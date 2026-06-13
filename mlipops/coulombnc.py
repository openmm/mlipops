import torch
from . import coulomb
from .neighborlist import NeighborList
from .pairwise import Pairwise
from .utils import periodic_displacements


class CoulombNC(torch.nn.Module):
    """Compute Coulomb interactions with no cutoff.

    This class computes the energy of a set of charges.  Because it directly calculates all interactions regardless of
    distance, it can only be used for non-periodic systems.  It is primarily useful for small, isolated molecules.

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
    def __init__(self, exclusions: torch.Tensor, prefactor: float, device: str = 'cpu'):
        """Create on object for computing Coulomb interactions.

        Parameters
        ----------
        exclusions: torch.Tensor
            a tensor of shape (n_exclusions, 2).  Each row contains the indices of two particles whose interaction
            should be omitted.
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0).  This sets the unit system.
        device: str
            the PyTorch device to perform calculation on.
        """
        if prefactor <= 0:
            raise ValueError('prefactor must be positive')
        super().__init__()
        self.neighbor_list = NeighborList(device=device)
        self.register_buffer('exclusions', exclusions)
        self.prefactor = prefactor
        self.pairwise = Pairwise(coulomb.point_charge_interaction, None, exclusions)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor):
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle

        Returns
        -------
        a torch.Tensor containing the energy of the interaction
        """
        neighbors = self.neighbor_list(positions, None)
        energy = self.pairwise(positions, charges, neighbors, None)
        return self.prefactor*energy

    def compute_field(self, field_positions: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor):
        """Compute the electric field produced by the particles at a set of points.

        Parameters
        ----------
        field_positions: torch.Tensor
            a Tensor of shape (n_points, 3) containing the positions at which to compute the field
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle

        Returns
        -------
        a Tensor of shape (n_points, 3) containing the electric field at each of the points
        """
        delta = periodic_displacements(field_positions.view((-1,1,3))-positions, None)
        r = torch.linalg.vector_norm(delta, dim=2, keepdim=True)
        field = charges.unsqueeze(1)*delta*r**-3
        field = torch.where((r > 0), field, 0)
        field = field.sum(dim=1)
        return self.prefactor*field
