import torch
import math
from .coulombnc import CoulombNC
from .coulombrf import CoulombRF
from .coulombewald import CoulombEwald
from .utils import pairwise_displacements


class ChargeEquilibration(torch.nn.Module):

    def __init__(self, coulomb: CoulombNC | CoulombRF | CoulombEwald):
        super().__init__()
        self.coulomb = coulomb

    def forward(self, positions: torch.Tensor, electronegativity: torch.Tensor, hardness: torch.Tensor,
                radius: torch.Tensor, total_charge: float = None, molecules: list = None,
                box_vectors: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.coulomb, CoulombNC):
            if box_vectors is not None:
                raise ValueError('Cannot use periodic boundary conditions with CoulombNC')
            interaction = self._compute_interactions_nc(positions, hardness, radius)
        elif isinstance(self.coulomb, CoulombRF):
            interaction = self._compute_interactions_rf(positions, hardness, radius, box_vectors)
        elif isinstance(self.coulomb, CoulombEwald):
            if box_vectors is None:
                raise ValueError('Must specify box_vectors for CoulombEwald')
            interaction = self._compute_interactions_ewald(positions, hardness, radius, box_vectors)
        else:
            raise ValueError('coulomb must be a CoulombNC, CoulombRF, or CoulombEwald')
        device = positions.device
        n = positions.shape[0]
        if total_charge is not None:
            if molecules is not None:
                raise ValueError('total_charge and molecules were both specified')
            constraint = torch.ones((n, 1), dtype=torch.float32, device=device)
            zeros = torch.zeros((1, 1), dtype=torch.float32, device=device)
            mol_charges = torch.tensor([total_charge], dtype=torch.float32, device=device)
        elif molecules is not None:
            m = len(molecules)
            constraint = torch.zeros((n, m), dtype=torch.float32, device=device)
            mol_charges = []
            for i, (indices, charge) in enumerate(molecules):
                constraint[indices, i] = 1
                mol_charges.append(charge)
            zeros = torch.zeros((m, m), dtype=torch.float32, device=device)
            mol_charges = torch.tensor(mol_charges, dtype=torch.float32, device=device)
        else:
            raise ValueError('Neither total_charge nor molecules was specified')
        matrix = torch.cat([torch.cat([interaction, constraint], dim=1),
                            torch.cat([constraint.T, zeros], dim=1)])
        x = torch.cat([-electronegativity, mol_charges])
        return torch.linalg.solve(matrix, x)[:n]

    def _compute_interactions_nc(self, positions: torch.Tensor, hardness: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        distance = torch.linalg.vector_norm(positions.view((-1,1,3)) - positions, dim=2)
        radius2 = radius**2
        gamma = torch.rsqrt(radius2.view((-1,1)) + radius2)
        return torch.where(torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device),
                           hardness + (math.sqrt(2/math.pi))/radius,
                           torch.erf(gamma*distance)/distance)

    def _compute_interactions_rf(self, positions: torch.Tensor, hardness: torch.Tensor, radius: torch.Tensor, box_vectors: torch.Tensor | None) -> torch.Tensor:
        n = positions.shape[0]
        interactions = torch.zeros((n, n), dtype=torch.float32, device=positions.device)
        pairs = self.coulomb.neighbor_list(positions, box_vectors)
        delta = pairwise_displacements(positions, pairs, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        radius2 = radius**2
        gamma = torch.rsqrt(radius2[pairs[:,0]] + radius2[pairs[:,0]])
        k = self.coulomb.pairwise.computation.k
        c = self.coulomb.pairwise.computation.c
        values = torch.erf(gamma*distance) * (1/distance + k*distance**2 - c)
        interactions[pairs[:,0], pairs[:,1]] = values
        interactions[pairs[:,1], pairs[:,0]] = values
        return torch.where(torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device),
                           hardness + (math.sqrt(2/math.pi))/radius,
                           interactions)

    def _compute_interactions_ewald(self, positions: torch.Tensor, hardness: torch.Tensor, radius: torch.Tensor, box_vectors: torch.Tensor | None) -> torch.Tensor:
        n = positions.shape[0]
        interactions = torch.zeros((n, n), dtype=torch.float32, device=positions.device)

        # Compute direct space interactions.

        pairs = self.coulomb.neighbor_list(positions, box_vectors)
        delta = pairwise_displacements(positions, pairs, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        radius2 = radius**2
        gamma = torch.rsqrt(radius2[pairs[:,0]] + radius2[pairs[:,0]])
        values = (torch.erf(gamma*distance) - torch.erf(self.coulomb.alpha*distance))/distance
        interactions[pairs[:,0], pairs[:,1]] = values
        interactions[pairs[:,1], pairs[:,0]] = values

        # Compute reciprocal space interactions.

        recip_box_vectors = torch.linalg.inv(box_vectors)
        k = self.coulomb.wave_indices@(2*torch.pi*recip_box_vectors.T)
        phase = k@positions.T
        cos = phase.cos()
        sin = phase.sin()
        k2 = (k*k).sum(dim=1)
        ak = torch.exp(self.coulomb._exp_coeff*k2)/k2
        interactions += torch.einsum('i,ij,ik->jk', ak, cos, cos) + torch.einsum('i,ij,ik->jk', ak, sin, sin)
        return torch.where(torch.eye(positions.shape[0], dtype=torch.bool, device=positions.device),
                           hardness + (math.sqrt(2/math.pi))/radius,
                           (4*torch.pi*recip_box_vectors.diag().prod())*interactions)
