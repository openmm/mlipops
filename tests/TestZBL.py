import torch
import pickle
import pytest
from mlipops import NeighborList, ZBL, get_covalent_radii


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_nonperiodic(device):
    """Test ZBL on a nonperiodic system."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    neighbor_list = NeighborList(None, device=device)
    zbl = ZBL(neighbor_list, 138.935, 0.052917721)
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    positions = torch.tensor(pos, dtype=torch.float32, device=device)
    numbers = torch.tensor([i+1 for i in range(9)], dtype=torch.int32, device=device)

    # Compare the energy to a value computed with TorchMD-Net.

    energy = zbl(positions, numbers, None, None)
    assert torch.allclose(torch.tensor(99.5888), energy)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_periodic(device):
    """Test ZBL on a periodic system."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    neighbor_list = NeighborList(None, device=device)
    zbl = ZBL(neighbor_list, 138.935, 0.052917721)
    pos = [[0.7713206433, 0.02075194936, 0.6336482349],
           [0.7488038825, 0.4985070123, 0.2247966455],
           [0.1980628648, 0.7605307122, 0.1691108366],
           [0.08833981417, 0.6853598184, 0.9533933462],
           [0.003948266328, 0.5121922634, 0.8126209617],
           [0.6125260668, 0.7217553174, 0.2918760682],
           [0.9177741225, 0.7145757834, 0.542544368],
           [0.1421700476, 0.3733407601, 0.6741336151],
           [0.4418331744, 0.4340139933, 0.6177669785]]
    positions = torch.tensor(pos, dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0, 1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    numbers = torch.tensor([i+1 for i in range(9)], dtype=torch.int32, device=device)

    # Compare the energy to a value computed with TorchMD-Net.

    energy = zbl(positions, numbers, None, box_vectors)
    assert torch.allclose(torch.tensor(109.3795), energy)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cutoff(device):
    """Test the cutoff function for ZBL."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    numbers = torch.tensor([10, 15], dtype=torch.int32, device=device)
    radii = torch.tensor([0.9, 0.8], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(3.0, device=device)
    zbl = ZBL(neighbor_list, 138.935, 0.052917721)
    for d in [0.1*i for i in range(5, 25)]:
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, d, 0.0]], dtype=torch.float32, device=device)
        energy1 = zbl(positions, numbers, None, None)
        energy2 = zbl(positions, numbers, radii, None)
        if d >= 1.7:
            assert torch.allclose(torch.tensor(0.0), energy2)
        else:
            assert torch.allclose(energy1*0.5*(torch.cos(torch.tensor(torch.pi*d/1.7, device=device))+1), energy2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_batch(device, periodic):
    """Test ZBL for a batch of systems."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_systems = 10
    num_particles = 20*num_systems
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    positions.requires_grad_()
    numbers = torch.randint(5, 10, (num_particles,), device=device)
    radii = get_covalent_radii(numbers, 0.052917721)
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
    zbl = ZBL(neighbor_list, 138.935, 0.052917721)
    energy = zbl(positions, numbers, radii, box_vectors, batch)
    for i in range(num_systems):
        mask = batch == i
        energy1 = energy[i]
        energy1.backward(retain_graph=True)
        grad1 = positions.grad[mask]
        pos = torch.tensor(positions[mask], device=device, requires_grad=True)
        box = None if box_vectors is None else box_vectors[i]
        energy2 = zbl(pos, numbers[mask], radii[mask], box)
        assert torch.allclose(energy1, energy2)
        energy2.backward()
        grad2 = pos.grad
        assert torch.allclose(grad1, grad2)
        positions.grad.zero_()
        pos.grad.zero_()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that ZBL can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    numbers = torch.tensor([i+1 for i in range(9)], dtype=torch.int32, device=device)
    radii = torch.rand((9,), dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(0.5, device=device)
    zbl = ZBL(neighbor_list, 138.935, 0.052917721)

    # Check that torch.compile works correctly.

    compiled = torch.compile(zbl)
    energy1 = zbl(positions, numbers, radii, box_vectors)
    energy2 = compiled(positions, numbers, radii, box_vectors)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(zbl)
    rf2 = pickle.loads(pickled)
    energy3 = rf2(positions, numbers, radii, box_vectors)
    assert torch.allclose(energy1, energy3)
