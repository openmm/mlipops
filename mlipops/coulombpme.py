import torch
import math
from . import coulomb
from .neighborlist import NeighborList
from .pairwise import Pairwise
try:
    import triton
    from .pme_triton import spread_charge_kernel, interp_derivatives_kernel
    has_triton = True
except ImportError:
    has_triton = False


class CoulombPME(torch.nn.Module):
    """Compute Coulomb interactions using the Particle Mesh Ewald method.

    This class computes the energy of an infinite set of charges repeating periodically through space.  The interaction
    is divided into a short range part, which is computed in direct space, and a long range part, which is computed
    in reciprocal space.  The division between the two is set by a parameter `alpha`, which can be adjusted to
    minimize the total cost of computing both parts.

    You can optionally specify that certain interactions should be omitted when computing the energy.  This is typically
    used for nearby atoms within the same molecule.  When two atoms are listed as an exclusion, only the interaction of
    each with the same periodic copy of the other (that is, not applying periodic boundary conditions) is excluded.
    Each atom still interacts with all the periodic copies of the other.

    Due to the way the reciprocal space term is calculated, it is impossible to prevent it from including excluded
    interactions.  The direct space term therefore compensates for it, subtracting off the energy that was incorrectly
    included in reciprocal space.  The sum of the two terms thus yields the correct energy with the interaction fully
    excluded.

    When performing backpropagation, this class computes derivatives with respect to atomic positions and charges, but
    not to any other parameters (box vectors, alpha, etc.).  In addition, it only computes first derivatives.
    Attempting to compute a second derivative will throw an exception.  This means that if you use PME during training,
    the loss function can only depend on energy, not forces.

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
    def __init__(self, neighbor_list: NeighborList, exclusions: torch.Tensor, gridx: int, gridy: int, gridz: int,
                 order: int, alpha: float, prefactor: float, cutoff: float | None = None):
        """Create on object for computing Coulomb interactions.

        Parameters
        ----------
        neighbor_list: NeighborList
            the NeighborList used to identify direct space interactions.  It determines the direct space cutoff
            distance, the device to run on, and whether padding is used to enable caching of neighbors.
        exclusions: torch.Tensor
            a tensor of shape (n_exclusions, 2).  Each row contains the indices of two particles whose interaction
            should be omitted.
        gridx: int
            the size of the charge grid along the x axis
        gridy: int
            the size of the charge grid along the y axis
        gridz: int
            the size of the charge grid along the z axis
        order: int
            the B-spline order to use for charge spreading
        alpha: float
            the coefficient of the erf() function used to separate the energy into direct and reciprocal space terms
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0).  This sets the unit system.
        cutoff: float | None
            the cutoff distance used when computing direct space interactions.  If None, the NeighborList's cutoff
            is used.  This argument is useful when a single NeighborList is shared by multiple interactions that use
            different cutoffs.  The value may never be greater than the NeighborList's cutoff.
        """
        if neighbor_list.include_self or neighbor_list.include_symmetric:
            raise ValueError('The neighbor list for Coulomb should not include self interactions or symmetric interactions')
        if gridx <= order or gridy <= order or gridz <= order:
            raise ValueError('The grid dimensions must be greater than the spline order')
        if order < 1:
            raise ValueError('order must be positive')
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
        self.gridx = gridx
        self.gridy = gridy
        self.gridz = gridz
        self.order = order
        self.alpha = alpha
        self.prefactor = prefactor
        self.cutoff = neighbor_list.cutoff if cutoff is None else cutoff
        self.use_triton = has_triton and torch.device(device).type == 'cuda'
        self.direct = Pairwise(coulomb.CoulombEwald(alpha), self.cutoff, exclusions)
        self.exclusion_correction = Pairwise(coulomb.CoulombEwaldExclusionCorrection(alpha), None)

        # Initialize the bspline moduli.

        max_size = max(gridx, gridy, gridz)
        data = torch.zeros(order, dtype=torch.float32, device=device)
        ddata = torch.zeros(order, dtype=torch.float32, device=device)
        bsplines_data = torch.zeros(max_size, dtype=torch.float32, device=device)
        data[0] = 1
        for i in range(3, order):
            data[i-1] = 0
            for j in range(1, i-1):
                data[i-j-1] = (j*data[i-j-2]+(i-j)*data[i-j-1])/(i-1)
            data[0] /= i-1

        # Differentiate.

        ddata[0] = -data[0]
        ddata[1:order] = data[0:order-1]-data[1:order]
        for i in range(1, order-1):
            data[order-i-1] = (i*data[order-i-2]+(order-i)*data[order-i-1])/(order-1)
        data[0] /= order-1
        bsplines_data[1:order+1] = data

        # Evaluate the actual bspline moduli for X/Y/Z.

        moduli = []
        for ndata in (gridx, gridy, gridz):
            m = torch.zeros(ndata, dtype=torch.float32, device=device)
            for i in range(ndata):
                arg = (2*torch.pi*i/ndata)*torch.arange(ndata, device=device)
                sc = torch.sum(bsplines_data[:ndata]*torch.cos(arg))
                ss = torch.sum(bsplines_data[:ndata]*torch.sin(arg))
                m[i] = sc*sc + ss*ss
            for i in range(ndata):
                if m[i] < 1e-7:
                    m[i] = (m[(i-1+ndata)%ndata]+m[(i+1)%ndata])*0.5
            moduli.append(m)
        self.xmoduli = torch.nn.Parameter(moduli[0], requires_grad=False)
        self.ymoduli = torch.nn.Parameter(moduli[1], requires_grad=False)
        self.zmoduli = torch.nn.Parameter(moduli[2], requires_grad=False)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor, include_direct: bool = True,
                include_reciprocal: bool = True):
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle
        box_vectors: torch.Tensor | None
            a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
            conditions are not used.
        include_direct: bool
            specifies whether the direct space term should be included in the result
        include_reciprocal: bool
            specifies whether the reciprocal space term should be included in the result

        Returns
        -------
        a torch.Tensor containing the energy of the interaction
        """
        energy = torch.zeros((1,), dtype=torch.float32, device=positions.device)
        if include_direct:
            neighbors = self.neighbor_list(positions, box_vectors)
            energy += self.direct(positions, charges, neighbors, box_vectors)
            if self.exclusions is not None:
                energy -= self.exclusion_correction(positions, charges, self.exclusions, None)
        if include_reciprocal:
            volume = box_vectors.diag().prod()
            energy -= torch.sum(charges**2)*self.alpha/math.sqrt(torch.pi)
            energy -= 0.5*torch.pi*torch.sum(charges)**2/(volume*self.alpha*self.alpha)
            energy += ReciprocalFunction.apply(self, positions, charges, box_vectors)
        return self.prefactor*energy


class ReciprocalFunction(torch.autograd.Function):
    """Compute the forward and backward passes of the reciprocal space interaction.  When possible, this uses Triton
    kernels to make the calculation faster.
    """
    @staticmethod
    def forward(ctx, pme: CoulombPME, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor):
        device = pme.xmoduli.device
        grid_size = (pme.gridx, pme.gridy, pme.gridz)
        grid_size_tensor = torch.tensor(grid_size, device=device)
        recip_box_vectors = torch.linalg.inv(box_vectors)
        t = torch.matmul(positions, recip_box_vectors)
        t = (t-torch.floor(t))*grid_size_tensor
        ti = t.to(torch.int32)
        dr = t-ti

        # Compute the B-spline coefficients.

        order = pme.order
        num_particles = positions.shape[0]
        data = torch.zeros((order, num_particles, 3), device=device)
        ddata = torch.zeros((order, num_particles, 3), device=device)
        data[order-1] = 0
        data[1] = dr
        data[0] = 1-dr
        for j in range(3, order):
            data[j-1] = dr*data[j-2]/(j-1)
            for k in range(1, j-1):
                data[j-k-1] = ((dr+k)*data[j-k-2]+(j-k-dr)*data[j-k-1])/(j-1)
            data[0] = (1-dr)*data[0]/(j-1)
        ddata[0] = -data[0]
        for j in range(1, order):
            ddata[j] = data[j-1]-data[j]
        data[order-1] = dr*data[order-2]/(order-1)
        for j in range(1, order-1):
            data[order-j-1] = ((dr+j)*data[order-j-2]+(order-j-dr)*data[order-j-1])/(order-1)
        data[0] = (1-dr)*data[0]/(order-1)

        # Spread charges on the grid.

        grid = torch.zeros(grid_size, dtype=torch.float32, device=device)
        if pme.use_triton:
            block_size = triton.next_power_of_2(order*order*order)
            spread_charge_kernel[lambda meta: (block_size,)](grid, grid_size_tensor, charges, data, ti, num_particles, order, block_size)
        else:
            spread_charge(grid, grid_size_tensor, charges, data, ti, order)

        # Take the Fourier transform, perform the convolution, and calculate the energy.

        recip_grid = torch.fft.rfftn(grid)
        scale_factor = torch.pi*box_vectors.diag().prod()
        recip_exp_factor = (torch.pi/pme.alpha)**2
        kx = torch.arange(recip_grid.shape[0], device=recip_grid.device)
        ky = torch.arange(recip_grid.shape[1], device=recip_grid.device)
        kz = torch.arange(recip_grid.shape[2], device=recip_grid.device)
        mx = (kx - (kx >= (pme.gridx+1)/2)*pme.gridx).view((-1,1,1)).expand(recip_grid.shape)
        my = (ky - (ky >= (pme.gridy+1)/2)*pme.gridy).view((1,-1,1)).expand(recip_grid.shape)
        mz = (kz - (kz >= (pme.gridz+1)/2)*pme.gridz).view((1,1,-1)).expand(recip_grid.shape)
        mhx = mx*recip_box_vectors[0,0]
        mhy = mx*recip_box_vectors[1,0] + my*recip_box_vectors[1,1]
        mhz = mx*recip_box_vectors[2,0] + my*recip_box_vectors[2,1] + mz*recip_box_vectors[2][2]
        m2 = mhx*mhx + mhy*mhy + mhz*mhz
        moduli = pme.xmoduli[kx].view((-1,1,1)) * pme.ymoduli[ky].view((1,-1,1)) * pme.zmoduli[kz].view((1,1,-1))
        denom = scale_factor*m2*moduli
        eterm = torch.exp(-recip_exp_factor*m2)/denom
        eterm[0,0,0] = 0
        scale = ((kz > 0)*(kz <= (pme.gridz-1)/2) + 1)
        energy = torch.sum(scale*eterm*(recip_grid.real*recip_grid.real + recip_grid.imag*recip_grid.imag))
        ctx.save_for_backward(positions, charges, recip_box_vectors, ti, data, ddata, recip_grid*eterm)
        ctx.pme = pme
        return 0.5*energy

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        positions, charges, recip_box_vectors, ti, data, ddata, recip_grid = ctx.saved_tensors
        pme = ctx.pme
        device = pme.xmoduli.device
        grid_size = (pme.gridx, pme.gridy, pme.gridz)
        grid_size_tensor = torch.tensor(grid_size, device=device)
        order = pme.order

        # Take the inverse Fourier transform.

        grid = torch.fft.irfftn(recip_grid, grid_size, norm='forward')

        # Compute the derivatives.

        num_particles = positions.shape[0]
        pos_deriv = torch.zeros_like(positions)
        charge_deriv = torch.zeros_like(charges)
        if pme.use_triton:
            g = lambda meta: (triton.cdiv(num_particles, meta['BLOCK_SIZE']),)
            interp_derivatives_kernel[g](pos_deriv, charge_deriv, grid, grid_size_tensor, data, ddata, ti, num_particles, order, 256)
        else:
            interp_derivatives(pos_deriv, charge_deriv, grid, grid_size_tensor, data, ddata, ti, order)
        pos_deriv = charges.view((-1,1))*torch.matmul(pos_deriv, (grid_size_tensor.view((1,3))*recip_box_vectors).T)
        return None, pos_deriv*grad_outputs[0], charge_deriv*grad_outputs[0], None

def spread_charge(grid: torch.Tensor, grid_size_tensor: torch.Tensor, charges: torch.Tensor,
                  data: torch.Tensor, ti: torch.Tensor, order: int):
    for ix in range(order):
        xindex = (ti[:,0]+ix) % grid_size_tensor[0]
        for iy in range(order):
            yindex = (ti[:,1]+iy) % grid_size_tensor[1]
            for iz in range(order):
                zindex = (ti[:,2]+iz) % grid_size_tensor[2]
                values = charges*data[ix,:,0]*data[iy,:,1]*data[iz,:,2]
                grid.index_put_((xindex, yindex, zindex), values, accumulate=True)
    return grid

def interp_derivatives(pos_deriv: torch.Tensor, charge_deriv: torch.Tensor, grid: torch.Tensor, grid_size_tensor: torch.Tensor,
                  data: torch.Tensor, ddata: torch.Tensor, ti: torch.Tensor, order: int):
    for ix in range(order):
        xindex = (ti[:,0]+ix) % grid_size_tensor[0]
        for iy in range(order):
            yindex = (ti[:,1]+iy) % grid_size_tensor[1]
            for iz in range(order):
                zindex = (ti[:,2]+iz) % grid_size_tensor[2]
                g = grid[xindex, yindex, zindex]
                pos_deriv[:,0] += ddata[ix,:,0]*data[iy,:,1]*data[iz,:,2]*g
                pos_deriv[:,1] += data[ix,:,0]*ddata[iy,:,1]*data[iz,:,2]*g
                pos_deriv[:,2] += data[ix,:,0]*data[iy,:,1]*ddata[iz,:,2]*g
                charge_deriv += data[ix,:,0]*data[iy,:,1]*data[iz,:,2]*g
