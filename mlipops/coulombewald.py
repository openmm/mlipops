import torch
import math
from . import coulomb
from .neighborlist import NeighborList
from .pairwise import Pairwise
from .utils import periodic_displacements


class CoulombEwald(torch.nn.Module):
    """Compute Coulomb interactions using Ewald summation.

    This class computes the energy of an infinite set of multipoles repeating periodically through space.  The
    interaction is divided into a short range part, which is computed in direct space, and a long range part, which is
    computed in reciprocal space.  The division between the two is set by a parameter `alpha`, which can be adjusted to
    minimize the total cost of computing both parts.

    By default, the multipoles are simply point charges.  You can also include dipoles by passing
    `max_multipole='dipole'` to the constructor.  In that case, you must provide dipole moments along with charges when
    invoking the module.

    You can optionally specify that certain interactions should be omitted when computing the energy.  This is typically
    used for nearby atoms within the same molecule.  When two atoms are listed as an exclusion, only the interaction of
    each with the same periodic copy of the other (that is, not applying periodic boundary conditions) is excluded.
    Each atom still interacts with all the periodic copies of the other.

    Due to the way the reciprocal space term is calculated, it is impossible to prevent it from including excluded
    interactions.  The direct space term therefore compensates for it, subtracting off the energy that was incorrectly
    included in reciprocal space.  The sum of the two terms thus yields the correct energy with the interaction fully
    excluded.

    In addition to calculating energy and forces, this class can compute the electric field at arbitrary points in
    space.  To do this, call compute_field().

    When you create an instance of this class, you must specify the value of Coulomb's constant 1/(4*pi*eps0).  Its
    value depends on the units used for energy and distance.  The value you specify thus sets the unit system.  Here are
    the values for some common units.

    kJ/mol, nm: 138.935457
    kJ/mol, A: 1389.35457
    kcal/mol, nm: 33.2063713
    kcal/mol, A: 332.063713
    eV, nm: 1.43996454
    eV, A: 14.3996454
    hartree, bohr: 1.0
    """
    def __init__(self, neighbor_list: NeighborList, exclusions: torch.Tensor, kmaxx: int, kmaxy: int, kmaxz: int,
                 alpha: float, prefactor: float, cutoff: float | None = None, max_multipole='charge'):
        """Create on object for computing Coulomb interactions.

        Parameters
        ----------
        neighbor_list: NeighborList
            the NeighborList used to identify direct space interactions.  It determines the direct space cutoff
            distance, the device to run on, and whether padding is used to enable caching of neighbors.
        exclusions: torch.Tensor
            a tensor of shape (n_exclusions, 2).  Each row contains the indices of two particles whose interaction
            should be omitted.
        kmaxx: int
            the index of the maximum wave vector in the x direction.  All vectors between -kmaxx and kmaxx are included.
        kmaxy: int
            the index of the maximum wave vector in the x direction.  All vectors between -kmaxy and kmaxy are included.
        kmaxz: int
            the index of the maximum wave vector in the x direction.  All vectors between -kmaxz and kmaxz are included.
        alpha: float
            the coefficient of the erf() function used to separate the energy into direct and reciprocal space terms
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0).  This sets the unit system.
        cutoff: float | None
            the cutoff distance used when computing direct space interactions.  If None, the NeighborList's cutoff
            is used.  This argument is useful when a single NeighborList is shared by multiple interactions that use
            different cutoffs.  The value may never be greater than the NeighborList's cutoff.
        max_multipole: str
            the maximum multipole order for each particle.  Allowed options are `'charge'` (point charges only) and
            `'dipole'` (charges and dipoles).
        """
        if neighbor_list.include_self or neighbor_list.include_symmetric:
            raise ValueError('The neighbor list for Coulomb should not include self interactions or symmetric interactions')
        if kmaxx < 1:
            raise ValueError('kmaxx must be positive')
        if kmaxy < 1:
            raise ValueError('kmaxy must be positive')
        if kmaxz < 1:
            raise ValueError('kmaxz must be positive')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if prefactor <= 0:
            raise ValueError('prefactor must be positive')
        if cutoff is not None and cutoff > neighbor_list.cutoff:
            raise ValueError("The cutoff cannot be larger than the NeighborList's cutoff")
        super().__init__()
        device = neighbor_list.device
        self.neighbor_list = neighbor_list
        self.register_buffer('exclusions', exclusions)
        self.kmaxx = kmaxx
        self.kmaxy = kmaxy
        self.kmaxz = kmaxz
        self.alpha = alpha
        self._exp_coeff = -1/(4*alpha*alpha)
        self.prefactor = prefactor
        self.cutoff = neighbor_list.cutoff if cutoff is None else cutoff
        self.max_multipole = max_multipole
        if max_multipole == 'charge':
            self.direct = Pairwise(coulomb.ErfcScaledInteraction(coulomb.point_charge_interaction, alpha), self.cutoff, exclusions)
            self.exclusion_correction = Pairwise(coulomb.ErfScaledInteraction(coulomb.point_charge_interaction, alpha), None)
        elif max_multipole == 'dipole':
            self.direct = Pairwise(coulomb.ErfcScaledDipoleInteraction(alpha), self.cutoff, exclusions)
            self.exclusion_correction = Pairwise(coulomb.ErfScaledDipoleInteraction(alpha), None)
        else:
            raise ValueError(f'Illegal value for max_multipole: {max_multipole}')

        # Compute the list of wave vector indices.  Because of symmetry, we only need to include half of them.

        nx = kmaxx+1
        ny = 2*kmaxy+1
        nz = 2*kmaxz+1
        index = torch.arange(nx*ny*nz, device=device)
        wave_indices = torch.stack([index//(ny*nz), (index//nz)%ny-kmaxy, index%nz-kmaxz], dim=1)
        mask = (wave_indices[:,0] > 0) | (wave_indices[:,1] > 0) | ((wave_indices[:,1] == 0) & (wave_indices[:,2] > 0))
        self.wave_indices = torch.nn.Parameter(wave_indices[mask].to(torch.float32), requires_grad=False)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor,
                include_direct: bool = True, include_reciprocal: bool = True, dipoles: torch.Tensor | None = None,
                batch: torch.Tensor | None = None):
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle
        box_vectors: torch.Tensor | None
            if batch is None, a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If batch is
            not None, a Tensor of shape (n_systems, 3, 3) containing the box vectors for each system.  If None, periodic
            boundary conditions are not used.
        include_direct: bool
            specifies whether the direct space term should be included in the result
        include_reciprocal: bool
            specifies whether the reciprocal space term should be included in the result
        dipoles: torch.Tensor | None
            a Tensor of shape (n_particles, 3) containing the dipole moment of each particle.  If max_multipole is
            'charge', this is ignored.
        batch: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the index of the system each particle belongs to.  This must be
            sorted in ascending order, and every system must contain at least one particle.  If None, the interaction
            is computed for a single system instead of a batch of systems.

        Returns
        -------
        a torch.Tensor containing the energy of the interaction.  If batch is None, this is a scalar containing the
        total energy.  Otherwise, it has shape (n_systems,) containing the energy of each system in the batch.
        """
        if batch is None:
            num_systems = 1
        else:
            num_systems = batch.max()+1
        energy = torch.zeros((num_systems,), dtype=torch.float32, device=positions.device)
        if include_direct:
            neighbors = self.neighbor_list(positions, box_vectors, batch)
            if self.max_multipole == 'charge':
                params = charges
            else:
                params = (charges, dipoles)
            energy += self.direct(positions, params, neighbors, box_vectors, batch)
            if self.exclusions is not None:
                energy -= self.exclusion_correction(positions, params, self.exclusions, None, batch)
        if include_reciprocal:
            if batch is None:
                volume = box_vectors.diag().prod()
                energy -= torch.sum(charges**2)*self.alpha/math.sqrt(torch.pi)
                energy -= 0.5*torch.pi*torch.sum(charges)**2/(volume*self.alpha*self.alpha)
                if self.max_multipole != 'charge':
                    energy -= (2/3)*torch.sum(dipoles*dipoles)*self.alpha**3/math.sqrt(torch.pi)
                energy += self._compute_recip_energy(positions, charges, dipoles, box_vectors)
            else:
                volume = torch.einsum('ijj->ij', box_vectors).prod(dim=1)
                sum_charges2 = torch.zeros_like(energy)
                sum_charges2.scatter_add_(0, batch, charges**2)
                energy -= sum_charges2*self.alpha/math.sqrt(torch.pi)
                sum_charges = torch.zeros_like(energy)
                sum_charges.scatter_add_(0, batch, charges)
                energy -= 0.5*torch.pi*sum_charges**2/(volume*self.alpha*self.alpha)
                if self.max_multipole != 'charge':
                    sum_dipoles2 = torch.zeros_like(energy)
                    sum_dipoles2.scatter_add_(0, batch, (dipoles**2).sum(axis=1))
                    energy -= (2/3)*sum_dipoles2*self.alpha**3/math.sqrt(torch.pi)
                energy += self._compute_recip_energy_batch(positions, charges, dipoles, box_vectors, batch, num_systems)
        return self.prefactor*energy

    def compute_field(self, field_positions: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor,
                      box_vectors: torch.Tensor, include_direct: bool = True, include_reciprocal: bool = True,
                      dipoles: torch.Tensor | None = None):
        """Compute the electric field produced by the particles at a set of points.

        Parameters
        ----------
        field_positions: torch.Tensor
            a Tensor of shape (n_points, 3) containing the positions at which to compute the field
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle
        box_vectors: torch.Tensor
            a Tensor of shape (3, 3) containing box vectors defining the periodic box.
        include_direct: bool
            specifies whether the direct space term should be included in the result
        include_reciprocal: bool
            specifies whether the reciprocal space term should be included in the result
        dipoles: torch.Tensor | None
            a Tensor of shape (n_particles, 3) containing the dipole moment of each particle.  If max_multipole is
            'charge', this is ignored.

        Returns
        -------
        a Tensor of shape (n_points, 3) containing the electric field at each of the points
        """
        if include_direct:
            delta = periodic_displacements(field_positions.view((-1,1,3))-positions, box_vectors)
            r = torch.linalg.vector_norm(delta, dim=2, keepdim=True)
            temp1 = 2*self.alpha/math.sqrt(math.pi)
            temp2 = 4*self.alpha**3/math.sqrt(math.pi)
            alphar = self.alpha*r
            rinv2 = r**-2
            expfactor = torch.exp(-alphar**2)
            b0 = torch.erfc(alphar)/r
            b1 = rinv2*(b0 + temp1*expfactor)
            b2 = rinv2*(3*b1 + temp2*expfactor)
            field = charges.unsqueeze(1)*delta*b1
            if self.max_multipole != 'charge':
                field += -b1*dipoles + b2*(dipoles.unsqueeze(0)*delta).sum(axis=2, keepdim=True)*delta
            field = torch.where((r > 0)*(r < self.cutoff), field, 0)
            field = field.sum(dim=1)
        else:
            field = torch.zeros_like(field_positions)
        if include_reciprocal:
            sum1, sum2, k, ak, recip_box_vectors = self._compute_recip_sums(positions, charges, dipoles, box_vectors)
            phase = k@field_positions.T
            temp = 8*torch.pi*recip_box_vectors.diag().prod()*ak.unsqueeze(1)*(sum1.unsqueeze(1)*torch.sin(phase) - sum2.unsqueeze(1)*torch.cos(phase))
            field += (temp.unsqueeze(2)*k.unsqueeze(1)).sum(dim=0)
        return self.prefactor*field


    def _compute_recip_sums(self, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor | None,
                            box_vectors: torch.Tensor):
        recip_box_vectors = torch.linalg.inv(box_vectors)
        k = self.wave_indices@(2*torch.pi*recip_box_vectors.T)
        phase = k@positions.T
        if self.max_multipole == 'charge':
            sum1 = (torch.cos(phase)*charges).sum(dim=1)
            sum2 = (torch.sin(phase)*charges).sum(dim=1)
        else:
            kd = k@dipoles.T
            sum1 = (torch.cos(phase)*charges - torch.sin(phase)*kd).sum(dim=1)
            sum2 = (torch.sin(phase)*charges + torch.cos(phase)*kd).sum(dim=1)
        k2 = (k*k).sum(dim=1)
        ak = torch.exp(self._exp_coeff*k2)/k2
        return sum1, sum2, k, ak, recip_box_vectors


    def _compute_recip_energy(self, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor | None,
                              box_vectors: torch.Tensor):
        sum1, sum2, _, ak, recip_box_vectors = self._compute_recip_sums(positions, charges, dipoles, box_vectors)
        energy = torch.sum(ak*(sum1**2 + sum2**2))
        return energy*4*torch.pi*recip_box_vectors.diag().prod()


    def _compute_recip_energy_batch(self, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor | None,
                                    box_vectors: torch.Tensor, batch: torch.Tensor | None, num_systems: int):
        recip_box_vectors = torch.linalg.inv(box_vectors)
        k = self.wave_indices.unsqueeze(0)@(2*torch.pi*recip_box_vectors.transpose(1, 2))
        phase = torch.einsum('ijk,ik->ji', k[batch], positions)
        scatter_index = batch.expand(self.wave_indices.shape[0], -1).T
        if self.max_multipole == 'charge':
            sum1 = torch.zeros((num_systems, self.wave_indices.shape[0]), dtype=torch.float32, device=positions.device)
            sum2 = torch.zeros((num_systems, self.wave_indices.shape[0]), dtype=torch.float32, device=positions.device)
            sum1.scatter_add_(0, scatter_index, (torch.cos(phase)*charges).T)
            sum2.scatter_add_(0, scatter_index, (torch.sin(phase)*charges).T)
        else:
            kd = torch.einsum('ijk,ik->ji', k[batch], dipoles)
            sum1 = torch.zeros((num_systems, self.wave_indices.shape[0]), dtype=torch.float32, device=positions.device)
            sum2 = torch.zeros((num_systems, self.wave_indices.shape[0]), dtype=torch.float32, device=positions.device)
            sum1.scatter_add_(0, scatter_index, (torch.cos(phase)*charges - torch.sin(phase)*kd).T)
            sum2.scatter_add_(0, scatter_index, (torch.sin(phase)*charges + torch.cos(phase)*kd).T)
        k2 = (k*k).sum(dim=2)
        ak = torch.exp(self._exp_coeff*k2)/k2
        energy = torch.sum(ak*(sum1**2 + sum2**2), dim=1)
        return energy*4*torch.pi*torch.diagonal(recip_box_vectors, dim1=1, dim2=2).prod(dim=1)
