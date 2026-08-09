import torch
import math

class ChargeEquilibration(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, positions: torch.Tensor, electronegativity: torch.Tensor, hardness: torch.Tensor,
                radius: torch.Tensor, total_charge: float = None, molecules: list = None) -> torch.Tensor:
        device = positions.device
        delta = positions.view((-1,1,3)) - positions
        distance = torch.linalg.vector_norm(delta, dim=2)
        radius2 = radius**2
        pair_radius = torch.sqrt(radius2.view((-1,1)) + radius2)
        n = positions.shape[0]
        interaction = torch.where(torch.eye(n, dtype=torch.bool, device=device),
                                  hardness + (math.sqrt(2/math.pi))/radius,
                                  torch.erf(distance/pair_radius)/distance)
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