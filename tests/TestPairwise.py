import torch
import pickle
import pytest
import random
from mlipops import NeighborList, Pairwise, periodic_displacements


def coulomb(pairs, r, delta, params):
    return params[pairs[:,0]]*params[pairs[:,1]]/r


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_coulomb(device, periodic):
    """Test computing the forces and energy for a Coulomb interaction."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    positions.requires_grad_()
    charges = 2.0*torch.rand(num_particles, dtype=torch.float32, device=device)-1.0
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
    delta = periodic_displacements(pos[:,0]-pos[:,1], box_vectors)
    distance = torch.linalg.norm(delta, dim=-1)
    mask = (distance < cutoff).to(torch.int32)
    positions.grad.zero_()
    charges.grad.zero_()
    expected_energy = torch.sum(mask*coulomb(index, distance, delta, charges))
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
    charges = 2.0*torch.rand(num_particles, dtype=torch.float32, device=device)-1.0
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
            delta = positions[i]-positions[j]
            distance = torch.linalg.norm(delta, dim=-1)
            if distance < cutoff and (i, j) not in exclusion_set and (j, i) not in exclusion_set:
                expected_energy += coulomb(torch.tensor([[i,j]], device=device), distance, delta, charges)
    assert torch.allclose(expected_energy, energy, rtol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('periodic', [True, False])
def test_batch(device, periodic):
    """Test computing the forces and energies for a batch of systems."""
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
    pairs = neighbor_list(positions, box_vectors, batch)
    pairwise = Pairwise(coulomb, cutoff)
    energy = pairwise(positions, charges, pairs, box_vectors, batch)
    for i in range(num_systems):
        mask = batch == i
        energy1 = energy[i]
        energy1.backward(retain_graph=True)
        grad1 = positions.grad[mask]
        pos = torch.tensor(positions[mask], device=device, requires_grad=True)
        box = None if box_vectors is None else box_vectors[i]
        pairs2 = neighbor_list(pos, box)
        energy2 = pairwise(pos, charges[mask], pairs2, box)
        assert torch.allclose(energy1, energy2, rtol=1e-4)
        energy2.backward()
        grad2 = pos.grad
        assert torch.allclose(grad1, grad2, rtol=1e-4)
        positions.grad.zero_()
        pos.grad.zero_()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that Pairwise can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_particles = 200
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    charges = 2.0*torch.rand(num_particles, dtype=torch.float32, device=device)-1.0
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
