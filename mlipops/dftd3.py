import torch
import os
from .neighborlist import NeighborList
from .pairwise import Pairwise
from .utils import pairwise_displacements, batch_pairwise_displacements
try:
    import triton
    from .dftd3_triton import compute_c6_kernel, backprop_c6_kernel
    has_triton = True
except ImportError:
    has_triton = False


class DFTD3(torch.nn.Module):
    """Compute the DFT-D3(BJ) dispersion potential.

    This is a highly accurate model for dispersion that depends only on positions and atomic numbers.  It supports
    all elements through Lawrencium (atomic number 103).  It is described in https://doi.org/10.1063/1.3382344 and
    https://doi.org/10.1002/jcc.21759.

    This class computes only the two body part of the potential.  The original publication also described a three body
    part, but recommended omitting it by default.

    The potential depends on three parameters called s8, a1, and a2.  Normally they are chosen to give the best
    accuracy when combined with a particular DFT functional.  The appropriate values will depend on how it is being
    used.  For example, if a model is trained to reproduce DFT data without a dispersion correction, it would be
    appropriate to add a DFT-D3(BJ) potential using the parameters for the DFT functional used to generate the training
    data.

    When performing backpropagation with Triton acceleration, this class can only compute first derivatives.  If you
    need second derivatives, specify need_second_derivatives=True to disable Triton acceleration.  This will make it
    slower, but allow for arbitrary derivatives.

    When you create an instance of this class, you must specify the values of Coulomb's constant 1/(4*pi*eps0) and the
    Bohr radius.  The values you specify set the unit system.  See the User Guide for the values in common unit systems.
    """
    def __init__(self, neighbor_list: NeighborList, s8: float, a1: float, a2: float, prefactor: float, bohr_radius: float):
        """Create on object for computing the DFT-D3(BJ) dispersion potential.

        Parameters
        ----------
        neighbor_list: NeighborList
            the NeighborList used to identify interactions.  It determines the cutoff distance, the device to run on,
            and whether padding is used to enable caching of neighbors.
        s8: float
            the scale factor for the 1/r**8 term in the potential
        a1: float
            the first parameter of the Becke-Johnson damping function (dimensionless)
        a2: float
            the second parameter of the Becke-Johnson damping function (measured in Bohr)
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0)
        bohr_radius: float
            the Bohr radius
        """
        if neighbor_list.include_self or neighbor_list.include_symmetric:
            raise ValueError('The neighbor list for DFTD3 should not include self interactions or symmetric interactions')
        if prefactor <= 0:
            raise ValueError('prefactor must be positive')
        if bohr_radius <= 0:
            raise ValueError('bohr_radius must be positive')
        super().__init__()
        self.neighbor_list = neighbor_list
        self.s8 = s8
        self.a1 = a1
        self.a2 = a2
        self.prefactor = prefactor
        self.bohr_radius = bohr_radius
        cn_ref, c6_ref, r4r2 = torch.load(os.path.join(os.path.dirname(__file__), 'dftd3_params.pt'))
        self.cn_ref = cn_ref.to(neighbor_list.device)
        self.c6_ref = c6_ref.to(neighbor_list.device)
        self.c8_scale = torch.sqrt(0.5*(torch.sqrt(torch.arange(len(r4r2)))*r4r2).to(neighbor_list.device))
        self.num_ref = torch.sum(self.cn_ref >= 0, dim=1)
        self.use_triton = has_triton and torch.device(neighbor_list.device).type == 'cuda'
        cutoff = None if neighbor_list.padding is None else neighbor_list.cutoff
        self.pairwise = Pairwise(DFTD3Calculator(s8, a1, a2, bohr_radius), cutoff, None)

    def forward(self, positions: torch.Tensor, atomic_numbers: torch.Tensor, covalent_radii: torch.Tensor,
                box_vectors: torch.Tensor | None = None, batch: torch.Tensor | None = None,
                need_second_derivatives: bool = False) -> torch.Tensor:
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        atomic_numbers: torch.Tensor
            a Tensor of shape (n_particles,) containing the atomic number of each particle
        covalent_radii: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the covalent radius of each particle.  These should be
            obtained by calling get_covalent_radii().
        box_vectors: torch.Tensor | None
            if batch is None, a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If batch is
            not None, a Tensor of shape (n_systems, 3, 3) containing the box vectors for each system.  If None, periodic
            boundary conditions are not used.
        batch: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the index of the system each particle belongs to.  This must be
            sorted in ascending order, and every system must contain at least one particle.  If None, the interaction
            is computed for a single system instead of a batch of systems.
        need_second_derivatives: bool
            whether you intend to compute second derivatives of the returned value.  This disables Triton acceleration.

        Returns
        -------
        torch.Tensor:
            a torch.Tensor containing the energy of the interaction.  If batch is None, this is a scalar containing the
            total energy.  Otherwise, it has shape (n_systems,) containing the energy of each system in the batch.
        """
        pairs = self.neighbor_list(positions, box_vectors, batch)
        if batch is None:
            delta = pairwise_displacements(positions, pairs, box_vectors)
        else:
            delta = batch_pairwise_displacements(positions, pairs, batch, box_vectors)
        r = torch.linalg.vector_norm(delta, dim=1)
        num_atoms = positions.shape[0]
        num_pairs = pairs.shape[0]

        # Compute coordination numbers.

        radius_sum = torch.sum(covalent_radii[pairs], dim=1)
        f = 1/(1+torch.exp(-16.0*((4/3)*radius_sum/r-1)))
        cn = torch.zeros((num_atoms,), dtype=torch.float32, device=positions.device)
        cn.scatter_add_(0, pairs[:,0], f)
        cn.scatter_add_(0, pairs[:,1], f)

        # Compute the C6 coefficient by interpolating reference values.

        z1 = atomic_numbers[pairs[:,0]]
        z2 = atomic_numbers[pairs[:,1]]
        if self.use_triton and not need_second_derivatives:
            c6 = C6Function.apply(self, pairs, atomic_numbers, cn)
        else:
            max_ref = torch.max(self.num_ref[atomic_numbers])
            sum_c6 = torch.zeros((num_pairs,), dtype=torch.float32, device=positions.device)
            sum_weight = torch.zeros((num_pairs,), dtype=torch.float32, device=positions.device)
            cn_ref1 = self.cn_ref[z1]
            cn_ref2 = self.cn_ref[z2]
            cn1 = cn[pairs[:,0]]
            cn2 = cn[pairs[:,1]]
            for i in range(max_ref):
                for j in range(max_ref):
                    mask = (cn_ref1[:,i] >= 0) * (cn_ref2[:,j] >= 0)
                    weight_ij = mask*torch.exp(-4*((cn1-cn_ref1[:,i])**2 + (cn2-cn_ref2[:,j])**2))
                    sum_weight += weight_ij
                    sum_c6 += weight_ij*self.c6_ref[z1, z2, i, j]
            c6 = sum_c6/sum_weight

        # Compute the energy.

        c8 = 3*self.c8_scale[z1]*self.c8_scale[z2]*c6
        energy = -self.pairwise(positions, (c6, c8), pairs, box_vectors, batch)
        return self.prefactor*energy


class DFTD3Calculator(object):
    """Compute the DFT-D3(BJ) energy.  This is a callable object designed for use with Pairwise."""

    def __init__(self, s8: float, a1: float, a2: float, bohr_radius: float):
        self.s8 = s8
        self.a1 = a1
        self.a2 = a2
        self.bohr_radius = bohr_radius

    def __call__(self, pairs, r, delta, params):
        c6, c8 = params
        r0 = torch.sqrt(c8/c6)
        f = self.a1*r0 + self.a2
        r_in_bohr = r/self.bohr_radius
        return c6/(r_in_bohr**6 + f**6) + self.s8*c8/(r_in_bohr**8 + f**8)


class C6Function(torch.autograd.Function):
    """Compute C6 for each pair using Triton."""

    @staticmethod
    def forward(ctx, d3: DFTD3, pairs: torch.Tensor, atomic_numbers: torch.Tensor, cn: torch.Tensor):
        num_pairs = pairs.shape[0]
        c6 = torch.empty((num_pairs,), dtype=torch.float32, device=atomic_numbers.device)
        g = lambda meta: (triton.cdiv(num_pairs, meta['BLOCK_SIZE']),)
        compute_c6_kernel[g](c6, atomic_numbers, cn, d3.cn_ref, d3.c6_ref, d3.num_ref, pairs, num_pairs, 256)
        ctx.save_for_backward(pairs, atomic_numbers, cn)
        ctx.d3 = d3
        return c6

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        pairs, atomic_numbers, cn = ctx.saved_tensors
        d3 = ctx.d3
        return C6BackwardFunction.apply(d3, pairs, atomic_numbers, cn, grad_outputs[0])


class C6BackwardFunction(torch.autograd.Function):
    """Compute the derivative of C6 for each pair using Triton."""

    @staticmethod
    def forward(ctx, d3: DFTD3, pairs: torch.Tensor, atomic_numbers: torch.Tensor, cn: torch.Tensor, grad_output: torch.Tensor):
        num_particles = atomic_numbers.shape[0]
        num_pairs = pairs.shape[0]
        result = torch.zeros((num_particles,), dtype=torch.float32, device=atomic_numbers.device)
        g = lambda meta: (triton.cdiv(num_pairs, meta['BLOCK_SIZE']),)
        backprop_c6_kernel[g](result, grad_output, atomic_numbers, cn, d3.cn_ref, d3.c6_ref, d3.num_ref, pairs, num_pairs, 256)
        return None, None, None, result

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        raise NotImplementedError('DFTD3 with Triton does not support second derivatives.  Specify need_second_derivatives=True to disable Triton acceleration.')
