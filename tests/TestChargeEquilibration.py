import torch
import math
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


def validate_minimization(coulomb, positions, charges, box_vectors, hardness, electronegativity, radius):
    e0 = coulomb(positions, charges, box_vectors) + (electronegativity*charges + 0.5*hardness*charges**2).sum()
    for _ in range(10):
        # Add a random offset to the charges and confirm that the energy increases.  This isn't strictly guaranteed,
        # since the minimization is based on Gaussian charges instead of point charges, so include a small margin.

        delta = 0.1*torch.randn_like(charges)
        delta -= torch.mean(delta)
        c2 = charges+delta
        e2 = coulomb(positions, c2, box_vectors) + (electronegativity*c2 + 0.5*(hardness+(math.sqrt(2/torch.pi)/radius))*c2**2).sum()
        assert e2 > e0-0.05


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
    validate_minimization(coulomb, positions, charges, None, hardness, electronegativity, radius)
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
    validate_minimization(coulomb, positions, charges, None, hardness, electronegativity, radius)
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
    cutoff = 3.0
    neighbor_list = NeighborList(cutoff, device=device)
    coulomb = CoulombEwald(neighbor_list, None, 12, 12, 12, 2.0, 1.0)
    eq = ChargeEquilibration(coulomb)
    box_vectors = torch.tensor([[9.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]], dtype=torch.float32, device=device)

    # Check that the charges are reasonable.

    charges = eq(positions, electronegativity, hardness, radius, 0, box_vectors=box_vectors)
    validate_minimization(coulomb, positions, charges, box_vectors, hardness, electronegativity, radius)
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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('method, periodic', [('nc', False), ('rf', False), ('rf', True), ('ewald', True)])
def test_qeq_derivatives(device, method, periodic):
    """Test computing derivatives of QEq charges."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions, electronegativity, hardness, radius = get_nh3_tensors(device)
    positions.requires_grad_(True)
    electronegativity.requires_grad_(True)
    hardness.requires_grad_(True)
    radius.requires_grad_(True)
    if method == 'nc':
        coulomb = CoulombNC(None, 1.0, device=device)
    elif method == 'rf':
        neighbor_list = NeighborList(2.0, device=device)
        coulomb = CoulombRF(neighbor_list, None, 1.0)
    else:
        neighbor_list = NeighborList(2.0, device=device)
        coulomb = CoulombEwald(neighbor_list, None, 7, 5, 5, 2.0, 1.0)
    eq = ChargeEquilibration(coulomb)
    if periodic:
        box_vectors = torch.tensor([[9.0, 0, 0], [0, 4.0, 0], [0, 0, 5.0]], dtype=torch.float32, device=device)
    else:
        box_vectors = None

    # Compute the charges and their derivatives.

    charges = eq(positions, electronegativity, hardness, radius, 0, box_vectors=box_vectors)
    result = (charges*charges).sum()
    pos_grad = torch.autograd.grad(result, positions, retain_graph=True)[0]
    elec_grad = torch.autograd.grad(result, electronegativity, retain_graph=True)[0]
    hardness_grad = torch.autograd.grad(result, hardness, retain_graph=True)[0]
    radius_grad = torch.autograd.grad(result, radius)[0]

    # Check them against a finite difference approximation.

    delta = 0.02
    norm = torch.linalg.norm(pos_grad)
    c1 = eq(positions+0.5*delta*pos_grad/norm, electronegativity, hardness, radius, 0, box_vectors=box_vectors)
    c2 = eq(positions-0.5*delta*pos_grad/norm, electronegativity, hardness, radius, 0, box_vectors=box_vectors)
    assert torch.allclose((c1*c1).sum() - (c2*c2).sum(), norm*delta, rtol=1e-2)
    norm = torch.linalg.norm(elec_grad)
    c1 = eq(positions, electronegativity+0.5*delta*elec_grad/norm, hardness, radius, 0, box_vectors=box_vectors)
    c2 = eq(positions, electronegativity-0.5*delta*elec_grad/norm, hardness, radius, 0, box_vectors=box_vectors)
    assert torch.allclose((c1*c1).sum() - (c2*c2).sum(), norm*delta, rtol=1e-2)
    norm = torch.linalg.norm(hardness_grad)
    c1 = eq(positions, electronegativity, hardness+0.5*delta*hardness_grad/norm, radius, 0, box_vectors=box_vectors)
    c2 = eq(positions, electronegativity, hardness-0.5*delta*hardness_grad/norm, radius, 0, box_vectors=box_vectors)
    assert torch.allclose((c1*c1).sum() - (c2*c2).sum(), norm*delta, rtol=1e-2)
    norm = torch.linalg.norm(radius_grad)
    c1 = eq(positions, electronegativity, hardness, radius+0.5*delta*radius_grad/norm, 0, box_vectors=box_vectors)
    c2 = eq(positions, electronegativity, hardness, radius-0.5*delta*radius_grad/norm, 0, box_vectors=box_vectors)
    assert torch.allclose((c1*c1).sum() - (c2*c2).sum(), norm*delta, rtol=1e-2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_qeq_external_potential(device):
    """Test the QEq algorithm with an external electric potential."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions, electronegativity, hardness, radius = get_nh3_tensors(device)
    coulomb = CoulombNC(None, 1.0, device=device)
    eq = ChargeEquilibration(coulomb)
    field = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    potential = -(positions*field).sum(axis=1)
    charges = eq(positions, electronegativity, hardness, radius, 0, potential=potential)
    assert torch.all(charges[[0,2,3,4]] > charges[[1,5,6,7]])
