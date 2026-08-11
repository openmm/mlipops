import torch
import pickle
import pytest
from mlipops import ChargeEquilibration, CoulombNC, CoulombRF, CoulombEwald, NeighborList


def get_nh3_tensors(device):
    """Get tensors describing the test system consisting of two NH3 molecules."""
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
    return positions, electronegativity, hardness, radius


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq_nc(device):
    """Test the QEq algorithm with no cutoff."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions, electronegativity, hardness, radius = get_nh3_tensors(device)
    coulomb = CoulombNC(None, 1.0, device=device)
    eq = ChargeEquilibration(coulomb)

    # Compare to charges computed with tad-multicharge.

    charges = eq(positions, electronegativity, hardness, radius, 0)
    assert torch.allclose(torch.tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731]), charges.cpu(), atol=1e-4)
    charges = eq(positions, electronegativity, hardness, radius, 1)
    assert torch.allclose(torch.tensor([-0.6708, -0.6708,  0.3982,  0.3745,  0.3982,  0.3982,  0.3745,  0.3982]), charges.cpu(), atol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq_rf(device):
    """Test the QEq algorithm with reaction field."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions, electronegativity, hardness, radius = get_nh3_tensors(device)
    cutoff = 20.0
    neighbor_list = NeighborList(cutoff, device=device)
    coulomb = CoulombRF(neighbor_list, None, 1.0)
    eq = ChargeEquilibration(coulomb)

    # Check that the charges are reasonable.

    charges = eq(positions, electronegativity, hardness, radius, 0)
    assert torch.allclose(charges.sum(), torch.tensor(0.0), atol=1e-4)
    assert torch.all(charges[:2] < 0.0)
    assert torch.all(charges[2:] > 0.0)
    charges = eq(positions, electronegativity, hardness, radius, 1)
    assert torch.allclose(charges.sum(), torch.tensor(1.0), atol=1e-4)

    # Translate one of the molecules so they are further apart than the cutoff.  The charges should be
    # identical to a single isolated molecule.

    positions2 = positions.detach().clone()
    index = [1, 5, 6, 7]
    positions2[index, 0] += 100.0
    charges1 = eq(positions2, electronegativity, hardness, radius, 0)
    charges2 = eq(positions[index], electronegativity[index], hardness[index], radius[index], 0)
    assert torch.allclose(charges1[index], charges2, atol=1e-4)
    assert torch.allclose(charges1[[0, 2, 3, 4]], charges2, atol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq_ewald(device):
    """Test the QEq algorithm with Ewald."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions, electronegativity, hardness, radius = get_nh3_tensors(device)
    cutoff = 2.0
    neighbor_list = NeighborList(cutoff, device=device)
    coulomb = CoulombEwald(neighbor_list, None, 7, 5, 5, 2.0, 1.0)
    eq = ChargeEquilibration(coulomb)
    box_vectors = torch.tensor([[9.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]], dtype=torch.float32, device=device)

    # Check that the charges are reasonable.

    charges = eq(positions, electronegativity, hardness, radius, 0, box_vectors=box_vectors)
    assert torch.allclose(charges.sum(), torch.tensor(0.0), atol=1e-4)
    assert torch.all(charges[:2] < 0.0)
    assert torch.all(charges[2:] > 0.0)
    charges = eq(positions, electronegativity, hardness, radius, 1, box_vectors=box_vectors)
    assert torch.allclose(charges.sum(), torch.tensor(1.0), atol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq_molecules(device):
    """Test the QEq algorithm with molecule charges."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions, electronegativity, hardness, radius = get_nh3_tensors(device)
    coulomb = CoulombNC(None, 1.0, device=device)
    eq = ChargeEquilibration(coulomb)

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
