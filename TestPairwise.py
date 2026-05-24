import torch
import pickle
import pytest
import random
from neighborlist import NeighborList
from pairwise import Pairwise

def compute_distances(delta, box_vectors):
    if box_vectors is not None:
        scale = torch.round(delta[:, 2]/box_vectors[2,2])
        delta -= scale.reshape((-1,1))*box_vectors[2].reshape((1,3))
        scale = torch.round(delta[:, 1]/box_vectors[1,1])
        delta -= scale.reshape((-1,1))*box_vectors[1].reshape((1,3))
        scale = torch.round(delta[:, 0]/box_vectors[0,0])
        delta -= scale.reshape((-1,1))*box_vectors[0].reshape((1,3))
    distance = torch.linalg.norm(delta, dim=-1)
    return distance

def coulomb(pairs, distance, parameters):
    return parameters[:,0,0]*parameters[:,1,0]/distance

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_coulomb(device, periodic):
    """Test computing the forces and energy for a Coulomb interaction."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    positions.requires_grad_()
    charges = 2.0*torch.rand((num_particles,1), dtype=torch.float32, device=device)-1.0
    charges.requires_grad_()
    if periodic:
        box_vectors = torch.tensor([[2.0, 0.0, 0.0],
                                    [0.1, 1.6, 0.0],
                                    [0.2, 0.1, 1.5]], dtype=torch.float32, device=device)
    else:
        box_vectors = None
    cutoff = 0.2
    neighbor_list = NeighborList(cutoff, device=device)
    pairs = neighbor_list(positions, box_vectors)
    pairwise = Pairwise(coulomb, cutoff)
    energy = pairwise(positions, charges, pairs, box_vectors)
    energy.backward()
    forces = -positions.grad
    grad = charges.grad
    index = torch.combinations(torch.arange(num_particles, device=device))
    pos = positions[index]
    distance = compute_distances(pos[:,0]-pos[:,1], box_vectors)
    mask = (distance < cutoff).to(torch.int32)
    positions.grad.zero_()
    charges.grad.zero_()
    expected_energy = torch.sum(mask*coulomb(pairs, distance, charges[index]))
    expected_energy.backward()
    assert torch.allclose(expected_energy, energy, rtol=1e-4)
    assert torch.allclose(-positions.grad, forces, rtol=1e-4)
    assert torch.allclose(charges.grad, grad, rtol=1e-4)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exclusions(device):
    """Test a force with excluded interactions."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    charges = 2.0*torch.rand((num_particles,1), dtype=torch.float32, device=device)-1.0
    cutoff = 0.4
    neighbor_list = NeighborList(cutoff, device=device)
    pairs = neighbor_list(positions, None)
    exclusion_set = set()
    for _ in range(10*num_particles):
        exclusion_set.add((random.randrange(num_particles), random.randrange(num_particles)))
    exclusions = torch.tensor(list(exclusion_set), dtype=torch.int32, device=device)
    pairwise = Pairwise(coulomb, cutoff, exclusions)
    energy = pairwise(positions, charges, pairs, None)
    expected_energy = 0
    for i in range(num_particles):
        for j in range(i):
            distance = compute_distances(positions[i]-positions[j], None)
            if distance < cutoff and (i, j) not in exclusion_set and (j, i) not in exclusion_set:
                expected_energy += coulomb(None, distance, torch.stack([charges[i], charges[j]]).unsqueeze(0))
    assert torch.allclose(expected_energy, energy, rtol=1e-4)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that Pairwise can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    charges = 2.0*torch.rand((num_particles,1), dtype=torch.float32, device=device)-1.0
    cutoff = 0.4
    neighbor_list = NeighborList(cutoff, device=device)
    pairs = neighbor_list(positions, None)
    pairwise = Pairwise(coulomb, cutoff)

    # Check that torch.compile works correctly.

    compiled = torch.compile(pairwise)
    energy1 = pairwise(positions, charges, pairs, None)
    energy2 = compiled(positions, charges, pairs, None)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(pairwise)
    pairwise2 = pickle.loads(pickled)
    energy3 = pairwise2(positions, charges, pairs, None)
    assert torch.allclose(energy1, energy3)
