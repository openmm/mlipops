import torch
import pickle
import pytest
from mlipops import NeighborList, CoulombNC


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_nonperiodic(device):
    """Test Coulomb on a nonperiodic system."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    coulomb = CoulombNC(None, 138.935, device=device)
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)

    # Compare forces and energies to values computed with OpenMM.

    energy = coulomb(positions, charges)
    assert torch.allclose(torch.tensor(-69.62032017036329), energy)
    expected_forces = [[-63.83682051937518, 63.7452128837408, 23.003853579542184],
                       [-45.77202235775695, 81.35381119054044, 85.72381514304159],
                       [6.971869046656136, -13.344767372827338, 35.71887727300647],
                       [11.913237226614143, -21.566292977262712, -20.630397577569504],
                       [0.0, 0.0, 0.0],
                       [16.59654830653884, -33.488046008760364, -46.82022735220012],
                       [20.660622675599516, -22.90393110011767, -30.743859279199974],
                       [-138.6858244351739, -24.863751423507416, 22.665277770003033],
                       [192.15239005689742, -28.932235191805745, -68.91733955662369]]
    energy.backward()
    assert torch.allclose(torch.tensor(expected_forces), -positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exclusions(device):
    """Test Coulomb with exclusions."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    excl = [[0, 3],
            [2, 3],
            [8, 7]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    coulomb = CoulombNC(exclusions, 138.935, device=device)

    # Compare forces and energies to values computed with OpenMM.

    energy = coulomb(positions, charges)
    assert torch.allclose(torch.tensor(-132.26861589686743), energy.cpu())
    expected_forces = [[-67.57396099141742, 67.38182003213029, 24.753437675041646],
                       [-45.77202235775695, 81.35381119054044, 85.72381514304159],
                       [6.366175813412341, -13.759725774458754, 40.048273933859974],
                       [16.256070931900183, -24.78794172402078, -26.70937833392247],
                       [0.0, 0.0, 0.0],
                       [16.59654830653884, -33.488046008760364, -46.82022735220012],
                       [20.660622675599516, -22.90393110011767, -30.743859279199974],
                       [27.571785578325795, 8.79867094838557, -8.607780032233958],
                       [25.894780043397706, -62.59465756369873, -37.64428175438669]]
    energy.backward()
    assert torch.allclose(torch.tensor(expected_forces), -positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compute_field(device):
    """Test computing the electric field."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((30, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(30)], dtype=torch.float32, device=device)
    field_positions = 3*torch.rand((10, 3), dtype=torch.float32, device=device)-1
    coulomb = CoulombNC(None, 138.935, device=device)
    field = coulomb.compute_field(field_positions, positions, charges)

    # Compare the field at each position to the force on a particle of charge 1 at the same position.

    for p, f1 in zip(field_positions, field):
        padded_pos = torch.cat([positions, p.unsqueeze(0)])
        padded_pos.requires_grad_(True)
        padded_charges = torch.nn.functional.pad(charges, pad=(0,1), value=1)
        energy = coulomb(padded_pos, padded_charges)
        energy.backward()
        f2 = -padded_pos.grad[-1]
        norm1 = torch.linalg.vector_norm(f1)
        norm2 = torch.linalg.vector_norm(f2)
        diffnorm = torch.linalg.vector_norm(f1-f2)/norm1
        assert torch.allclose(norm1, norm2, rtol=5e-3)
        assert diffnorm < 5e-3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that CoulombNC can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    coulomb = CoulombNC(None, 138.935, device=device)

    # Check that torch.compile works correctly.

    compiled = torch.compile(coulomb)
    energy1 = coulomb(positions, charges)
    energy2 = compiled(positions, charges)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(coulomb)
    coulomb2 = pickle.loads(pickled)
    energy3 = coulomb2(positions, charges)
    assert torch.allclose(energy1, energy3)
