import torch
from . import coulomb
from .neighborlist import NeighborList
from .pairwise import Pairwise
from .utils import periodic_displacements


class CoulombRF(torch.nn.Module):
    """Compute Coulomb interactions using the reaction field approximation.

    This class computes the energy of a set of charges, either with or without periodic boundary conditions.
    Interactions beyond a cutoff distance are ignored, and the form of the potential is modified based on two
    assumptions: first, that everything beyond the cutoff is filled with bulk solvent, and second, that bulk solvent
    can be modeled as a uniform dielectric.

    You can optionally specify that certain interactions should be omitted when computing the energy.  This is typically
    used for nearby atoms within the same molecule.

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
    def __init__(self, neighbor_list: NeighborList, exclusions: torch.Tensor, prefactor: float,
                 dielectric: float = 78.3, cutoff: float | None = None):
        """Create on object for computing Coulomb interactions.

        Parameters
        ----------
        neighbor_list: NeighborList
            the NeighborList used to identify interactions.  It determines the cutoff distance, the device to run on,
            and whether padding is used to enable caching of neighbors.
        exclusions: torch.Tensor
            a tensor of shape (n_exclusions, 2).  Each row contains the indices of two particles whose interaction
            should be omitted.
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0).  This sets the unit system.
        dielectric: float
            the dielectric constant to use for the solvent.  The default value corresponds to water.
        cutoff: float | None
            the cutoff distance used when computing space interactions.  If None, the NeighborList's cutoff is used.
            This argument is useful when a single NeighborList is shared by multiple interactions that use different
            cutoffs.  The value may never be greater than the NeighborList's cutoff.
        """
        if neighbor_list.include_self or neighbor_list.include_symmetric:
            raise ValueError('The neighbor list for Coulomb should not include self interactions or symmetric interactions')
        if prefactor <= 0:
            raise ValueError('prefactor must be positive')
        if dielectric <= 0:
            raise ValueError('dielectric must be positive')
        if cutoff is not None and cutoff > neighbor_list.cutoff:
            raise ValueError("The cutoff cannot be larger than the NeighborList's cutoff")
        super().__init__()
        self.neighbor_list = neighbor_list
        self.register_buffer('exclusions', exclusions)
        self.prefactor = prefactor
        self.dielectric = dielectric
        self.cutoff = neighbor_list.cutoff if cutoff is None else cutoff
        self.pairwise = Pairwise(coulomb.ReactionFieldInteraction(self.cutoff, dielectric), self.cutoff, exclusions)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor):
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle
        box_vectors: torch.Tensor
            a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
            conditions are not used.

        Returns
        -------
        a torch.Tensor containing the energy of the interaction
        """
        neighbors = self.neighbor_list(positions, box_vectors)
        energy = self.pairwise(positions, charges, neighbors, box_vectors)
        return self.prefactor*energy

    def compute_field(self, field_positions: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor,
                      box_vectors: torch.Tensor):
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
            a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
            conditions are not used.

        Returns
        -------
        a Tensor of shape (n_points, 3) containing the electric field at each of the points
        """
        delta = periodic_displacements(field_positions.view((-1,1,3))-positions, box_vectors)
        r = torch.linalg.vector_norm(delta, dim=2, keepdim=True)
        k = self.pairwise.computation.k
        field = charges.unsqueeze(1)*delta*(1/r**3 - 2*k)
        field = torch.where((r > 0)*(r < self.cutoff), field, 0)
        field = field.sum(dim=1)
        return self.prefactor*field
