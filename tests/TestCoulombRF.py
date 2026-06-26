import torch
import pickle
import pytest
from mlipops import NeighborList, CoulombRF


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_nonperiodic(device):
    """Test reaction field on a nonperiodic system."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    rf = CoulombRF(neighbor_list, None, 138.935)
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

    energy = rf(positions, charges, None)
    assert torch.allclose(torch.tensor(5.279248480659704), energy)
    expected_forces = [[0.0, 0.0, 0.0],
                       [-16.403598601078414, 50.17931158347308, 27.033087486341337],
                       [5.050501343051991, -0.4725031862171719, 1.4959735003830055],
                       [1.2208592265194345, -7.076531567911882, -6.333556397300791],
                       [0.0, 0.0, 0.0],
                       [14.108640132450379, -36.17568551265617, -22.898637420760796],
                       [-0.888954539485777, -10.384567738143462, -9.19415819530517],
                       [-128.27081025114555, -18.647458530177563, 30.231653223316115],
                       [125.18336268968794, 22.57743495163318, -20.334362196673702]]
    energy.backward()
    assert torch.allclose(torch.tensor(expected_forces), -positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_periodic(device):
    """Test reaction field on a periodic system."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    rf = CoulombRF(neighbor_list, None, 138.935)
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
    box_vectors = torch.tensor([[1, 0, 0], [0, 1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)

    # Compare forces and energies to values computed with OpenMM.

    energy = rf(positions, charges, box_vectors)
    assert torch.allclose(torch.tensor(5.296856311773031), energy)
    expected_forces = [[6.168991485924394, -17.109168886119946, -3.8375256245936216],
                       [-16.403598601078414, 50.17931158347308, 27.033087486341337],
                       [3.466229579874882, -0.06836980849089247, 10.13052536728161],
                       [-1.682897174005643, -7.670103967398974, -15.08600644157537],
                       [0.0, 0.0, 0.0],
                       [14.108640132450379, -36.17568551265617, -22.898637420760796],
                       [-11.568913179834432, 20.59865755166617, -10.515884671536789],
                       [-119.2718149330191, -32.33207591210644, 35.508803501517335],
                       [125.18336268968794, 22.57743495163318, -20.334362196673702]]
    energy.backward()
    assert torch.allclose(torch.tensor(expected_forces), -positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exclusions(device):
    """Test reaction field exclusions."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
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
    neighbor_list = NeighborList(cutoff, device=device)
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0, 1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    rf = CoulombRF(neighbor_list, exclusions, 138.935)

    # Compare forces and energies to values computed with OpenMM.

    energy = rf(positions, charges, box_vectors)
    assert torch.allclose(torch.tensor(-5.113604855798104), energy.cpu())
    expected_forces = [[6.168991485924394, -17.109168886119946, -3.8375256245936216],
                       [-16.403598601078414, 50.17931158347308, 27.033087486341337],
                       [2.1923252979038685, -0.941117357356672, 5.303970800396397],
                       [-0.4089928920346291, -6.7973564185331945, -10.259451874690157],
                       [0.0, 0.0, 0.0],
                       [14.108640132450379, -36.17568551265617, -22.898637420760796],
                       [-11.568913179834432, 20.59865755166617, -10.515884671536789],
                       [7.778136091607013, -6.608085814016995, 11.610706675502007],
                       [-1.8665883349381773, -3.146555146456264, 3.5637346293416243]]
    energy.backward()
    assert torch.allclose(torch.tensor(expected_forces), -positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_batch(device, periodic):
    """Test Coulomb for a batch of systems."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_systems = 10
    num_particles = 20*num_systems
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    positions.requires_grad_()
    charges = 2.0*torch.rand(num_particles, dtype=torch.float32, device=device)-1.0
    batch = torch.arange(num_systems, device=device).expand((20,-1)).T.flatten()
    if periodic:
        box_vectors = []
        for i in range(num_systems):
            scale = 0.9+0.2*torch.rand(1, dtype=torch.float32, device=device)
            box_vectors.append(torch.tensor([[2.0, 0.0, 0.0],
                                             [0.1, 1.6, 0.0],
                                             [0.2, 0.1, 1.5]], dtype=torch.float32, device=device)*scale)
        box_vectors = torch.stack(box_vectors)
    else:
        box_vectors = None
    cutoff = 0.4
    neighbor_list = NeighborList(cutoff, device=device)
    coulomb = CoulombRF(neighbor_list, None, 138.935)
    energy = coulomb(positions, charges, box_vectors, batch)
    for i in range(num_systems):
        mask = batch == i
        energy1 = energy[i]
        energy1.backward(retain_graph=True)
        grad1 = positions.grad[mask]
        pos = torch.tensor(positions[mask], device=device, requires_grad=True)
        box = None if box_vectors is None else box_vectors[i]
        energy2 = coulomb(pos, charges[mask], box)
        assert torch.allclose(energy1, energy2)
        energy2.backward()
        grad2 = pos.grad
        assert torch.allclose(grad1, grad2)
        positions.grad.zero_()
        pos.grad.zero_()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compute_field(device):
    """Test computing the electric field."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((30, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(30)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    field_positions = 3*torch.rand((10, 3), dtype=torch.float32, device=device)-1
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    rf = CoulombRF(neighbor_list, None, 138.935)
    field = rf.compute_field(field_positions, positions, charges, box_vectors)

    # Compare the field at each position to the force on a particle of charge 1 at the same position.

    for p, f1 in zip(field_positions, field):
        padded_pos = torch.cat([positions, p.unsqueeze(0)])
        padded_pos.requires_grad_(True)
        padded_charges = torch.nn.functional.pad(charges, pad=(0,1), value=1)
        energy = rf(padded_pos, padded_charges, box_vectors)
        energy.backward()
        f2 = -padded_pos.grad[-1]
        norm1 = torch.linalg.vector_norm(f1)
        norm2 = torch.linalg.vector_norm(f2)
        diffnorm = torch.linalg.vector_norm(f1-f2)/norm1
        assert torch.allclose(norm1, norm2, rtol=5e-3)
        assert diffnorm < 5e-3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that CoulombRF can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(0.5, device=device)
    rf = CoulombRF(neighbor_list, None, 138.935)

    # Check that torch.compile works correctly.

    compiled = torch.compile(rf)
    energy1 = rf(positions, charges, box_vectors)
    energy2 = compiled(positions, charges, box_vectors)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(rf)
    rf2 = pickle.loads(pickled)
    energy3 = rf2(positions, charges, box_vectors)
    assert torch.allclose(energy1, energy3)
