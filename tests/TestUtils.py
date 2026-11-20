import torch
import pytest
from mlipops import NeighborList, create_supercell, batch_create_supercell, compute_supercell_neighbors, batch_compute_supercell_neighbors
from collections import defaultdict

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_supercell(device):
    """Test tiling a unit cell to create a supercell."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 7
    positions = 1.1*torch.rand((num_particles, 3), dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1.1, 0.0, 0.0],
                                [0.1, 1.6, 0.0],
                                [0.2, 0.1, 2.1]], dtype=torch.float32, device=device)
    cutoff = 1.9
    new_positions, new_box_vectors, particle, offset = create_supercell(positions, box_vectors, cutoff)
    expected_copies = torch.tensor([4, 3, 2], device=device)
    expected_total = int(expected_copies.prod())
    assert new_positions.shape[0] == num_particles*expected_total
    assert particle.shape[0] == num_particles*expected_total
    assert offset.shape[0] == num_particles*expected_total
    assert torch.allclose(expected_copies.unsqueeze(1)*box_vectors, new_box_vectors)
    for i in range(len(particle)):
        assert int(particle[i]) == i%num_particles
        cell = i//num_particles
        x = cell//(expected_copies[1]*expected_copies[2])
        y = (cell//expected_copies[2])%expected_copies[1]
        z = cell%expected_copies[2]
        assert torch.all(offset[i] == torch.tensor([x, y, z]))
        assert torch.allclose(new_positions[i], positions[i%num_particles]+x*box_vectors[0]+y*box_vectors[1]+z*box_vectors[2])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batch_supercell(device):
    """Test creating supercells for a batch of systems."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = [5, 4, 3]
    num_systems = len(num_particles)
    positions = 1.1*torch.rand((sum(num_particles), 3), dtype=torch.float32, device=device)
    box_vectors = torch.tensor([
                                [[3.9, 0.0, 0.0],
                                 [0.0, 1.6, 0.0],
                                 [0.0, 0.0, 1.1]],
                                [[1.1, 0.0, 0.0],
                                 [0.1, 1.6, 0.0],
                                 [0.2, 0.1, 2.1]],
                                [[1.5, 0.0, 0.0],
                                 [0.0, 1.1, 0.0],
                                 [0.0, 0.0, 0.9]]], dtype=torch.float32, device=device)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
    cutoff = 1.9
    new_positions, new_box_vectors, particle, offset, new_batch = batch_create_supercell(positions, box_vectors, cutoff, batch)
    expected_copies = torch.tensor([[1, 3, 4], [4, 3, 2], [3, 4, 5]], device=device)
    expected_total_copies = [int(expected_copies[i].prod()) for i in range(3)]
    expected_total_particles = sum(num_particles[i]*expected_total_copies[i] for i in range(3))
    expected_batch = torch.cat([i*torch.ones(num_particles[i]*expected_total_copies[i], dtype=torch.int32) for i in range(3)])
    assert new_positions.shape[0] == expected_total_particles
    assert particle.shape[0] == expected_total_particles
    assert offset.shape[0] == expected_total_particles
    for i in range(num_systems):
        assert torch.allclose(expected_copies[i].unsqueeze(1)*box_vectors[i], new_box_vectors[i])
    assert torch.allclose(expected_batch, new_batch)
    unitcell_start_index = 0
    supercell_start_index = 0
    for s in range(num_systems):
        for i in range(num_particles[s]*expected_total_copies[s]):
            assert int(particle[supercell_start_index+i]) == unitcell_start_index + i%num_particles[s]
            cell = i//num_particles[s]
            x = cell//(expected_copies[s][1]*expected_copies[s][2])
            y = (cell//expected_copies[s][2])%expected_copies[s][1]
            z = cell%expected_copies[s][2]
            assert torch.all(offset[supercell_start_index+i] == torch.tensor([x, y, z]))
            assert torch.allclose(new_positions[supercell_start_index+i], positions[unitcell_start_index+i%num_particles[s]]+x*box_vectors[s][0]+y*box_vectors[s][1]+z*box_vectors[s][2])
        unitcell_start_index += num_particles[s]
        supercell_start_index += num_particles[s]*expected_total_copies[s]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('include_self,include_symmetric', [(False,False),(False,True),(True,False),(True,True)])
def test_supercell_neighbors(device, include_self, include_symmetric):
    """Test computing neighbors for a supercell."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 7
    positions = 1.1*torch.rand((num_particles, 3), dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1.1, 0.0, 0.0],
                                [0.1, 1.6, 0.0],
                                [0.2, 0.1, 2.1]], dtype=torch.float32, device=device)
    cutoff = 1.9
    new_positions, new_box_vectors, points, offset = create_supercell(positions, box_vectors, cutoff)
    neighbor_list = NeighborList(cutoff, include_self, include_symmetric, device=device)
    neighbors1 = compute_supercell_neighbors(neighbor_list, new_positions, box_vectors, new_box_vectors, points, offset)
    neighbors2 = neighbor_list(new_positions, new_box_vectors)

    # Since there are 24 unit cells in the supercell, the full supercell should have 24 equivalent copies of each pair.

    count1 = defaultdict(int)
    count2 = defaultdict(int)
    for i in range(neighbors1.shape[0]):
        p1 = int(points[neighbors1[i, 0]])
        p2 = int(points[neighbors1[i, 1]])
        count1[(min(p1, p2), max(p1, p2))] += 1
    for i in range(neighbors2.shape[0]):
        p1 = int(points[neighbors2[i, 0]])
        p2 = int(points[neighbors2[i, 1]])
        count2[(min(p1, p2), max(p1, p2))] += 1
    for i in range(num_particles):
        for j in range(i, num_particles):
            assert count2[(i, j)] == 24*count1[(i, j)]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('include_self,include_symmetric', [(False,False),(False,True),(True,False),(True,True)])
def test_batch_supercell_neighbors(device, include_self, include_symmetric):
    """Test computing neighbors for a batch of supercells."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 12
    positions = 1.1*torch.rand((num_particles, 3), dtype=torch.float32, device=device)
    box_vectors = torch.tensor([
                                [[3.9, 0.0, 0.0],
                                 [0.0, 1.6, 0.0],
                                 [0.0, 0.0, 1.1]],
                                [[1.1, 0.0, 0.0],
                                 [0.1, 1.6, 0.0],
                                 [0.2, 0.1, 2.1]],
                                [[1.5, 0.0, 0.0],
                                 [0.0, 1.1, 0.0],
                                 [0.0, 0.0, 0.9]]], dtype=torch.float32, device=device)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2], device=device)
    copies = [12, 24, 60]
    cutoff = 1.9
    new_positions, new_box_vectors, points, offset, new_batch = batch_create_supercell(positions, box_vectors, cutoff, batch)
    neighbor_list = NeighborList(cutoff, include_self, include_symmetric, device=device)
    neighbors1 = batch_compute_supercell_neighbors(neighbor_list, new_positions, box_vectors, new_box_vectors, points, offset, new_batch)
    neighbors2 = neighbor_list(new_positions, new_box_vectors, batch=new_batch)

    # For a system with N unit cells in the supercell, the full supercell should have N equivalent copies of each pair.

    count1 = defaultdict(int)
    count2 = defaultdict(int)
    for i in range(neighbors1.shape[0]):
        p1 = int(points[neighbors1[i, 0]])
        p2 = int(points[neighbors1[i, 1]])
        count1[(min(p1, p2), max(p1, p2))] += 1
    for i in range(neighbors2.shape[0]):
        p1 = int(points[neighbors2[i, 0]])
        p2 = int(points[neighbors2[i, 1]])
        count2[(min(p1, p2), max(p1, p2))] += 1
    for i in range(num_particles):
        for j in range(i, num_particles):
            assert count2[(i, j)] == copies[batch[i]]*count1[(i, j)]
