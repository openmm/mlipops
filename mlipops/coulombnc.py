import torch
from . import coulomb
from .neighborlist import NeighborList
from .pairwise import Pairwise
from .utils import periodic_displacements


class CoulombNC(torch.nn.Module):
    """Compute Coulomb interactions with no cutoff.

    This class computes the energy of a set of multipoles.  Because it directly calculates all interactions regardless
    of distance, it can only be used for non-periodic systems.  It is primarily useful for small, isolated molecules.

    By default, the multipoles are simply point charges.  You can also include dipoles by passing
    `max_multipole='dipole'` to the constructor.  In that case, you must provide dipole moments along with charges when
    invoking the module.

    You can optionally specify that certain interactions should be omitted when computing the energy.  This is typically
    used for nearby atoms within the same molecule.

    In addition to calculating energy and forces, this class can compute the electric field at arbitrary points in
    space.  To do this, call compute_field().

    When you create an instance of this class, you must specify the value of Coulomb's constant 1/(4*pi*eps0).  Its
    value depends on the units used for energy and distance.  The value you specify thus sets the unit system.  See the
    User Guide for the values in common unit systems.
    """
    def __init__(self, exclusions: torch.Tensor, prefactor: float, max_multipole='charge', device: str = 'cpu'):
        """Create on object for computing Coulomb interactions.

        Parameters
        ----------
        exclusions: torch.Tensor
            a tensor of shape (n_exclusions, 2).  Each row contains the indices of two particles whose interaction
            should be omitted.
        prefactor: float
            Coulomb's constant 1/(4*pi*eps0).  This sets the unit system.
        max_multipole: str
            the maximum multipole order for each particle.  Allowed options are `'charge'` (point charges only) and
            `'dipole'` (charges and dipoles).
        device: str
            the PyTorch device to perform calculation on.
        """
        if prefactor <= 0:
            raise ValueError('prefactor must be positive')
        super().__init__()
        self.neighbor_list = NeighborList(device=device)
        self.register_buffer('exclusions', exclusions)
        self.prefactor = prefactor
        self.max_multipole = max_multipole
        if max_multipole == 'charge':
            computation = coulomb.point_charge_interaction
        elif max_multipole == 'dipole':
            computation = coulomb.dipole_interaction
        else:
            raise ValueError(f'Illegal value for max_multipole: {max_multipole}')
        self.pairwise = Pairwise(computation, None, exclusions)

    def forward(self, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor | None = None,
                batch: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges: torch.Tensor
            a Tensor of shape (n_particles,) containing the charge of each particle
        dipoles: torch.Tensor | None
            a Tensor of shape (n_particles, 3) containing the dipole moment of each particle.  If max_multipole is
            'charge', this is ignored.
        batch: torch.Tensor | None
            a Tensor of shape (n_particles,) containing the index of the system each particle belongs to.  This must be
            sorted in ascending order, and every system must contain at least one particle.  If None, the interaction
            is computed for a single system instead of a batch of systems.

        Returns
        -------
        torch.Tensor:
            a torch.Tensor containing the energy of the interaction.  If batch is None, this is a scalar containing the
            total energy.  Otherwise, it has shape (n_systems,) containing the energy of each system in the batch.
        """
        neighbors = self.neighbor_list(positions, None, batch)
        if self.max_multipole == 'charge':
            params = charges
        else:
            params = (charges, dipoles)
        energy = self.pairwise(positions, params, neighbors, None, batch)
        return self.prefactor*energy

    def compute_field(self, field_positions: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor,
                      dipoles: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the electric field produced by the particles at a set of points.

        Parameters
        ----------
        field_positions: torch.Tensor
            a Tensor of shape (n_points, 3) containing the positions at which to compute the field
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the Cartesian coordinates of each particle
        charges:
            a Tensor of shape (n_particles,) containing the charge of each particle
        dipoles: torch.Tensor | None
            a Tensor of shape (n_particles, 3) containing the dipole moment of each particle.  If max_multipole is
            'charge', this is ignored.

        Returns
        -------
        torch.Tensor:
            a Tensor of shape (n_points, 3) containing the electric field at each of the points
        """
        delta = periodic_displacements(field_positions.view((-1,1,3))-positions, None)
        r = torch.linalg.vector_norm(delta, dim=2, keepdim=True)
        denom3 = r**-3
        field = charges.unsqueeze(1)*delta*denom3
        if self.max_multipole != 'charge':
            field += (3*(dipoles.unsqueeze(0)*delta).sum(axis=2, keepdim=True)*delta*r**-2 - dipoles)*denom3
        field = torch.where((r > 0), field, 0)
        field = field.sum(dim=1)
        return self.prefactor*field
