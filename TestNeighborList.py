import torch
import pickle
import pytest
import numpy as np
from neighborlist import NeighborList

def apply_pbc(delta, box_vectors):
    if box_vectors is not None:
        scale = torch.round(delta[:, 2]/box_vectors[2,2])
        delta -= scale.reshape((-1,1))*box_vectors[2].reshape((1,3))
        scale = torch.round(delta[:, 1]/box_vectors[1,1])
        delta -= scale.reshape((-1,1))*box_vectors[1].reshape((1,3))
        scale = torch.round(delta[:, 0]/box_vectors[0,0])
        delta -= scale.reshape((-1,1))*box_vectors[0].reshape((1,3))
    distance = torch.linalg.norm(delta, dim=-1)
    return distance

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
    neighbors = neighbor_list(positions, box_vectors)
    neighbors = neighbors.detach().cpu().numpy()

    # Check that all the returned neighbors are correct.

    for index in range(neighbors.shape[1]):
        i = neighbors[index, 0]
        j = neighbors[index, 1]
        if not include_self:
            assert i != j
        distance = apply_pbc((positions[i]-positions[j]).reshape((1,3)), box_vectors)
        assert distance <= cutoff

    # Check that the right number of neighbors was found.

    found = set(tuple(pair) for pair in neighbors)
    num_expected = 0
    index = torch.combinations(torch.arange(num_particles, device=device), with_replacement=include_self)
    pos = positions[index]
    distance = apply_pbc(pos[:,0]-pos[:,1], box_vectors)
    mask = (distance < cutoff).to(torch.int32)
    if include_symmetric:
        mask *= torch.where(index[:,0] != index[:,1], 2, 1)
    num_expected = torch.sum(mask)
    assert num_expected == len(found)

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
