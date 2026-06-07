import torch
import pickle
import pytest
from mlipops import NeighborList, periodic_displacements


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic,include_self,include_symmetric', [(False,False,False),(True,False,False),(False,False,True),(True,True,False),(False,True,True)])
def test_neighbors(device, periodic, include_self, include_symmetric):
    """Test that neighbor lists are computed correctly."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200 if device == 'cpu' else 1100
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    if periodic:
        box_vectors = torch.tensor([[2.0, 0.0, 0.0],
                                    [0.1, 1.6, 0.0],
                                    [0.2, 0.1, 1.5]], dtype=torch.float32, device=device)
    else:
        box_vectors = None
    cutoff = 0.2
    neighbor_list = NeighborList(cutoff, include_self, include_symmetric, device=device)
    assert neighbor_list.cutoff == cutoff
    assert neighbor_list.include_self == include_self
    assert neighbor_list.include_symmetric == include_symmetric
    neighbors = neighbor_list(positions, box_vectors)
    neighbors = neighbors.detach().cpu().numpy()

    # Check that all the returned neighbors are correct.

    for index in range(neighbors.shape[0]):
        i = int(neighbors[index, 0])
        j = int(neighbors[index, 1])
        if not include_self:
            assert i != j
        distance = torch.linalg.norm(periodic_displacements((positions[i]-positions[j]).reshape((1,3)), box_vectors), dim=-1)
        assert distance <= cutoff

    # Check that the right number of neighbors was found.

    found = set(tuple(pair) for pair in neighbors)
    index = torch.combinations(torch.arange(num_particles, device=device), with_replacement=include_self)
    pos = positions[index]
    distance = torch.linalg.norm(periodic_displacements(pos[:,0]-pos[:,1], box_vectors), dim=-1)
    mask = (distance < cutoff).to(torch.int32)
    if include_symmetric:
        mask *= torch.where(index[:,0] != index[:,1], 2, 1)
    num_expected = torch.sum(mask)
    assert num_expected == len(found)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('include_self', [True, False])
@pytest.mark.parametrize('include_symmetric', [True, False])
def test_no_cutoff(device, include_self, include_symmetric):
    """Test a neighbor list that does not use a cutoff."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 100
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    neighbor_list = NeighborList(None, include_self, include_symmetric, device=device)
    neighbors = neighbor_list(positions, None)
    if include_self:
        if include_symmetric:
            assert neighbors.shape[0] == num_particles*num_particles
        else:
            assert neighbors.shape[0] == (num_particles+1)*num_particles//2
    else:
        if include_symmetric:
            assert neighbors.shape[0] == num_particles*(num_particles-1)
        else:
            assert neighbors.shape[0] == (num_particles-1)*num_particles//2
    for index in range(neighbors.shape[0]):
        i = int(neighbors[index, 0])
        j = int(neighbors[index, 1])
        if not include_self:
            assert i != j
        if not include_symmetric:
            assert i <= j


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_padding(device):
    """Test padding when building a neighbor list."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200 if device == 'cpu' else 1100
    cutoff = 1.0
    padding = 0.2
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    neighbor_list = NeighborList(cutoff, padding=padding, device=device)
    assert neighbor_list.padding == padding
    neighbors1 = neighbor_list(positions, None)

    # Check that the right number of neighbors was found.

    found = set(tuple(pair) for pair in neighbors1)
    index = torch.combinations(torch.arange(num_particles, device=device))
    pos = positions[index]
    distance = torch.linalg.norm(pos[:,0]-pos[:,1], dim=-1)
    mask = (distance < cutoff+padding).to(torch.int32)
    num_expected = torch.sum(mask)
    assert num_expected == len(found)

    # Displacing the particles by a small amount should not change the return value.

    neighbors2 = neighbor_list(positions+0.5*padding*torch.rand((num_particles,3), dtype=torch.float32, device=device), None)
    assert neighbors1 is neighbors2

    # Displacing them by a larger amount should cause the neighbor list to be recalculated.

    neighbors3 = neighbor_list(positions+1.5*padding*torch.rand((num_particles,3), dtype=torch.float32, device=device), None)
    assert not neighbors1.equal(neighbors3)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_caching(device, periodic):
    """Test caching of neighbors."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200
    cutoff = 0.5
    if periodic:
        box_vectors = torch.tensor([[2.0, 0.0, 0.0],
                                    [0.1, 1.6, 0.0],
                                    [0.2, 0.1, 1.5]], dtype=torch.float32, device=device)
    else:
        box_vectors = None
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    neighbor_list = NeighborList(cutoff, device=device)
    neighbors1 = neighbor_list(positions, box_vectors)
    assert neighbor_list(positions, box_vectors) is neighbors1
    if periodic:
        assert neighbor_list(positions, box_vectors.clone()) is neighbors1
    neighbors2 = neighbor_list(positions.clone(), box_vectors)
    assert neighbors2 is not neighbors1
    assert neighbors2.equal(neighbors1)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that NeighborList can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200 if device == 'cpu' else 1200
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    neighbor_list = NeighborList(1.0, False, False, device=device)

    # Check that torch.compile works correctly.

    compiled = torch.compile(neighbor_list)
    neighbors1 = neighbor_list(positions, None)
    neighbors2 = compiled(positions, None)
    assert neighbors1.shape == neighbors2.shape

    # Check that pickle works correctly.

    pickled = pickle.dumps(neighbor_list)
    neighbor_list2 = pickle.loads(pickled)
    neighbors2 = neighbor_list2(positions, None)
    assert neighbors1.shape == neighbors2.shape
