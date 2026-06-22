import torch
from .neighborlist import NeighborList
from .pairwise import Pairwise


class ZBL(torch.nn.Module):
    """Compute the repulsive Ziegler-Biersack-Littmark (ZBL) potential between atoms.

    This is an empirical potential to describe screened nuclear repulsion between atoms.  It depends only on the atomic
    numbers of the interacting atoms, and gives a good fit for a wide range of elements.

    Because it was parameterized based on scattering data, it does not give an accurate description of the interaction
    between covalently bonded atoms.  It therefore should usually be restricted to only very short distances.  You can
    optionally provide a covalent radius for each atom.  The ZBL interaction is then multiplied by a cosine cutoff
    function to make it go to zero when the distance between two atoms equals the sum of their covalent radii.  You can
    obtain covalent radii by calling get_covalent_radii(), or you can use other values that you choose.

    When you create an instance of this class, you must specify the values of Coulomb's constant 1/(4*pi*eps0) and the
    Bohr radius.  The values you specify set the unit system.  Here are their values for some common units.

    kJ/mol, nm: 138.935457, 0.052917721
    kJ/mol, A: 1389.35457, 0.52917721
    kcal/mol, nm: 33.2063713, 0.052917721
    kcal/mol, A: 332.063713, 0.52917721
    eV, nm: 1.43996454, 0.052917721
    eV, A: 14.3996454, 0.52917721
    hartree, bohr: 1.0, 1.0
    """
    def __init__(self, neighbor_list: NeighborList, prefactor: float, bohr_radius: float):
        """Create on object for computing ZBL interactions.

        Parameters
        ----------
        neighbor_list: NeighborList
            the NeighborList used to identify interactions.  It determines the cutoff distance, the device to run on,
            and whether padding is used to enable caching of neighbors.
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0)
        bohr_radius: float
            the Bohr radius
        """
        if neighbor_list.include_self or neighbor_list.include_symmetric:
            raise ValueError('The neighbor list for ZBL should not include self interactions or symmetric interactions')
        if prefactor <= 0:
            raise ValueError('prefactor must be positive')
        if bohr_radius <= 0:
            raise ValueError('bohr_radius must be positive')
        super().__init__()
        self.neighbor_list = neighbor_list
        self.prefactor = prefactor
        self.bohr_radius = bohr_radius
        self.pairwise = Pairwise(ZBLCalculator(bohr_radius), None, None)

    def forward(self, positions: torch.Tensor, atomic_numbers: torch.Tensor, radii: torch.Tensor | None,
                box_vectors: torch.Tensor):
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        atomic_numbers: torch.Tensor
            a Tensor of shape (n_particles,) containing the atomic number of each particle
        radii: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the covalent radius of each particle.  If None, the cutoff
            function is omitted.
        box_vectors: torch.Tensor
            a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
            conditions are not used.

        Returns
        -------
        a torch.Tensor containing the energy of the interaction
        """
        neighbors = self.neighbor_list(positions, box_vectors)
        energy = self.pairwise(positions, (atomic_numbers, radii), neighbors, box_vectors)
        return self.prefactor*energy


class ZBLCalculator(object):
    """Compute the ZBL interaction.  This is a callable object designed for use with Pairwise."""

    def __init__(self, bohr_radius: float):
        self.bohr_radius = bohr_radius

    def __call__(self, pairs, r, delta, params):
        atomic_numbers, radii = params
        z = atomic_numbers[pairs]
        a = 0.8854*self.bohr_radius/torch.sum(z**0.23, dim=1)
        d = r/a
        f = 0.1818*torch.exp(-3.2*d) + 0.5099*torch.exp(-0.9423*d) + 0.2802*torch.exp(-0.4029*d) + 0.02817*torch.exp(-0.2016*d)
        energy = f*torch.prod(z, dim=1)/r
        if radii is not None:
            cutoff = torch.sum(radii[pairs], dim=1)
            energy *= torch.where(r < cutoff, 0.5*(torch.cos(torch.pi*r/cutoff)+1), 0.0)
        return energy
