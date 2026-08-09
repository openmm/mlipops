import torch
import pickle
import pytest
from mlipops import ChargeEquilibration


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq(device):
    """Test the QEq algorithm."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    eq = ChargeEquilibration()
    pos = [[-2.98334550857544, -0.08808205276728, 0.0],
           [2.98334550857544, 0.08808205276728, 0.0],
           [-4.07920360565186, 0.25775116682053, 1.52985656261444],
           [-1.60526800155640, 1.24380481243134, 0.0],
           [-4.07920360565186, 0.25775116682053, -1.52985656261444],
           [4.07920360565186, -0.25775116682053, -1.52985656261444],
           [1.60526800155640, -1.24380481243134, 0.0],
           [4.07920360565186, -0.25775116682053, 1.52985656261444]]
    positions = torch.tensor(pos, dtype=torch.float32, device=device)
    electronegativity = torch.tensor([1.3974671539272585]*2 + [1.18778931]*6, dtype=torch.float32, device=device)
    hardness = torch.tensor([0.05317918]*2 + [-0.35015861]*6, dtype=torch.float32, device=device)
    radius = torch.tensor([1.32250290]*2 + [0.55159092]*6, dtype=torch.float32, device=device)

    # Compare to charges computed with tad-multicharge.

    charges = eq(positions, electronegativity, hardness, radius, 0)
    assert torch.allclose(torch.tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731]), charges.cpu(), atol=1e-4)
    charges = eq(positions, electronegativity, hardness, radius, 1)
    assert torch.allclose(torch.tensor([-0.6708, -0.6708,  0.3982,  0.3745,  0.3982,  0.3982,  0.3745,  0.3982]), charges.cpu(), atol=1e-4)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq_molecules(device):
    """Test the QEq algorithm with molecule charges."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    eq = ChargeEquilibration()
    pos = [[-2.98334550857544, -0.08808205276728, 0.0],
           [2.98334550857544, 0.08808205276728, 0.0],
           [-4.07920360565186, 0.25775116682053, 1.52985656261444],
           [-1.60526800155640, 1.24380481243134, 0.0],
           [-4.07920360565186, 0.25775116682053, -1.52985656261444],
           [4.07920360565186, -0.25775116682053, -1.52985656261444],
           [1.60526800155640, -1.24380481243134, 0.0],
           [4.07920360565186, -0.25775116682053, 1.52985656261444]]
    positions = torch.tensor(pos, dtype=torch.float32, device=device)
    electronegativity = torch.tensor([1.3974671539272585]*2 + [1.18778931]*6, dtype=torch.float32, device=device)
    hardness = torch.tensor([0.05317918]*2 + [-0.35015861]*6, dtype=torch.float32, device=device)
    radius = torch.tensor([1.32250290]*2 + [0.55159092]*6, dtype=torch.float32, device=device)

    # With each molecule neutral, the charges should match the results from a global constraint.

    molecules = [(torch.tensor([0, 2, 3, 4], device=device), 0),
                (torch.tensor([1, 5, 6, 7], device=device), 0)]
    charges = eq(positions, electronegativity, hardness, radius, molecules=molecules)
    assert torch.allclose(torch.tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731]), charges.cpu(), atol=1e-4)

    # Try giving each molecule a different charge.

    molecules = [(torch.tensor([0, 2, 3, 4], device=device), 0),
                (torch.tensor([1, 5, 6, 7], device=device), 1)]
    charges = eq(positions, electronegativity, hardness, radius, molecules=molecules)
    assert torch.allclose(charges[molecules[0][0]].sum(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(charges[molecules[1][0]].sum(), torch.tensor(1.0), atol=1e-4)
