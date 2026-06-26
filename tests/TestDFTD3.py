import torch
import pickle
import pytest
from mlipops import NeighborList, DFTD3, get_covalent_radii


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_nonperiodic(device):
    """Test DFTD3 on a nonperiodic system."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    neighbor_list = NeighborList(device=device)
    d3 = DFTD3(neighbor_list, 0.78981345, 0.49484001, 5.73083694, 138.935, 0.052917721)
    pos = [[-0.0748, -0.0015, 0.0024],
           [0.0558, 0.042, -0.0278],
           [0.0716, 0.1404, 0.0137],
           [-0.1293, -0.0202, -0.0901],
           [-0.1263, 0.0754, 0.06],
           [-0.0699, -0.0934, 0.0609],
           [-0.5844, 0.1432, 0.3239],
           [-0.5605, -0.3153, 0.1213]]
    positions = torch.tensor(pos, dtype=torch.float32, device=device)
    positions.requires_grad_(True)
    numbers = torch.tensor([6, 8, 1, 1, 1, 1, 11, 17], dtype=torch.int32, device=device)
    radii = get_covalent_radii(numbers, 0.052917721)

    # Compare the energy and forces to values computed with tad-dftd3.

    energy = d3(positions, numbers, radii, None)
    assert torch.allclose(torch.tensor(-0.12475485), energy)
    energy.backward()
    expected_grad = [[0.06107452, 0.00206157, -0.03069437],
                     [0.02080633, 0.02562851, 0.00282612],
                     [0.01481178, -0.01370186, -0.01326269],
                     [0.02183147, -0.00062286, -0.01917054],
                     [0.04949829, -0.00073406, -0.02608646],
                     [0.03155244, -0.00446477, -0.01144230],
                     [-0.14291400, 0.19881181, 0.15526232],
                     [-0.05666087, -0.20697834, -0.05743211]]
    assert torch.allclose(torch.tensor(expected_grad), positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_batch(device, periodic):
    """Test DFTD3 for a batch of systems."""
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
    d3 = DFTD3(neighbor_list, 0.78981345, 0.49484001, 5.73083694, 138.935, 0.052917721)
    energy = d3(positions, numbers, radii, box_vectors, batch)
    for i in range(num_systems):
        mask = batch == i
        energy1 = energy[i]
        energy1.backward(retain_graph=True)
        grad1 = positions.grad[mask]
        pos = torch.tensor(positions[mask], device=device, requires_grad=True)
        box = None if box_vectors is None else box_vectors[i]
        energy2 = d3(pos, numbers[mask], radii[mask], box)
        assert torch.allclose(energy1, energy2)
        energy2.backward()
        grad2 = pos.grad
        assert torch.allclose(grad1, grad2)
        positions.grad.zero_()
        pos.grad.zero_()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that DFTD3 can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    numbers = torch.arange(9, device=device)+1
    radii = get_covalent_radii(numbers, 0.052917721)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(device=device)
    d3 = DFTD3(neighbor_list, 0.78981345, 0.49484001, 5.73083694, 138.935, 0.052917721)

    # Check that torch.compile works correctly.

    compiled = torch.compile(d3)
    energy1 = d3(positions, numbers, radii, box_vectors)
    energy2 = compiled(positions, numbers, radii, box_vectors)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(d3)
    d32 = pickle.loads(pickled)
    energy3 = d32(positions, numbers, radii, box_vectors)
    assert torch.allclose(energy1, energy3)
