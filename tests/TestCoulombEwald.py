import torch
import pickle
import pytest
from mlipops import NeighborList, CoulombEwald, CoulombPME


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rectangular(device):
    """Test Ewald on a rectangular box."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, None, 4, 4, 6, 5.25652, 138.935, cutoff)
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

    edirect = ewald(positions, charges, box_vectors, True, False)
    assert torch.allclose(torch.tensor(0.3928169131284689), edirect, rtol=1e-4)
    erecip = ewald(positions, charges, box_vectors, False, True)
    assert torch.allclose(torch.tensor(-90.76642523584681), erecip, rtol=1e-4)
    etotal = ewald(positions, charges, box_vectors)
    assert torch.allclose(etotal, edirect+erecip, rtol=1e-4)
    expected_ddirect = [[-0.24798475679812249, 0.687764457922038, 0.15426311754405225],
                        [7.054380497802695, -12.71947379932531, -4.4184430554814975],
                        [-0.16711740757171428, 0.0005975071737236055, -0.40411180885070197],
                        [0.05735632613468174, 0.3694616002473111, 0.6623167408218675],
                        [-0.0, -0.0, -0.0],
                        [-6.825540819092257, 12.065082692400136, 4.253138003397518],
                        [0.40846758855145704, -0.7899422147127474, 0.4264155026074473],
                        [24.33210191617286, 5.2702563627006604, -5.18767668229398],
                        [-24.6116633451996, -4.88374660640581, 4.514098182255295]]
    expected_drecip = [[-0.73552396935075, -27.411619319772516, -3.6772893441672396],
                       [32.02866446017598, -29.486999049219637, -83.37086148751101],
                       [-15.171483299948418, 10.333167442031195, -38.68395873797307],
                       [-7.400541832757025, 22.19981716737497, 40.2835867591183],
                       [-0.0, -0.0, -0.0],
                       [-14.14217722823658, 8.491757924755209, 35.12992818377489],
                       [19.810004939553213, -60.087083612855636, 34.233151271330215],
                       [128.8233026169692, 61.85932968401663, -28.915104743734883],
                       [-143.21224568640557, 14.101629763670012, 45.00054809916274]]
    edirect.backward()
    assert torch.allclose(torch.tensor(expected_ddirect), positions.grad.cpu(), rtol=1e-4, atol=1e-5)
    positions.grad.zero_()
    erecip.backward()
    assert torch.allclose(torch.tensor(expected_drecip), positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exclusions(device):
    """Test Ewald with exclusions."""
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
    box_vectors = torch.tensor([[1, 0, 0], [0, 1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, exclusions, 4, 4, 6, 5.25652, 138.935, cutoff)

    # Compare forces and energies to values computed with OpenMM.

    edirect = ewald(positions, charges, box_vectors, True, False)
    assert torch.allclose(torch.tensor(-62.26296411784335), edirect)
    erecip = ewald(positions, charges, box_vectors, False, True)
    assert torch.allclose(torch.tensor(-90.76642523584681), erecip)
    expected_ddirect = [[3.489155715227115, -2.948842690450898, -1.5953209779474455],
                        [7.054380497802695, -12.71947379932531, -4.4184430554814975],
                        [0.4920053058109548, 0.4521602663603476, -4.53107462237761],
                        [-4.338906859273225, 3.5545059894336237, 6.538863649840273],
                        [-0.0, -0.0, -0.0],
                        [-6.825540819092257, 12.065082692400136, 4.253138003397518],
                        [0.40846758855145704, -0.7899422147127474, 0.4264155026074473],
                        [-141.92550809732688, -28.392166009192334, 26.085381119943012],
                        [141.64594666830013, 28.778675765487183, -26.7589596199817]]
    expected_drecip = [[-0.73552396935075, -27.411619319772516, -3.6772893441672396],
                       [32.02866446017598, -29.486999049219637, -83.37086148751101],
                       [-15.171483299948418, 10.333167442031195, -38.68395873797307],
                       [-7.400541832757025, 22.19981716737497, 40.2835867591183],
                       [-0.0, -0.0, -0.0],
                       [-14.14217722823658, 8.491757924755209, 35.12992818377489],
                       [19.810004939553213, -60.087083612855636, 34.233151271330215],
                       [128.8233026169692, 61.85932968401663, -28.915104743734883],
                       [-143.21224568640557, 14.101629763670012, 45.00054809916274]]
    edirect.backward()
    assert torch.allclose(torch.tensor(expected_ddirect), positions.grad.cpu(), rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert torch.allclose(torch.tensor(expected_drecip), positions.grad.cpu(), rtol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('max_multipole', ['charge', 'dipole'])
def test_compare_to_pme(device, max_multipole):
    """Check the Ewald and PME agree with each other."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((30, 3), dtype=torch.float32, device=device)-1
    positions.requires_grad_(True)
    charges = 0*torch.tensor([(i-4)*0.1 for i in range(30)], dtype=torch.float32, device=device)
    dipoles = 0.05*(torch.rand((30, 3), dtype=torch.float32, device=device)-0.5)
    box_vectors = torch.tensor([[1, 0, 0], [0.1, 1.1, 0], [-0.1, 0.2, 1.2]], dtype=torch.float32, device=device)
    exclusions = torch.tensor([[0, 3], [2, 3], [8, 7]], dtype=torch.int32, device=device)
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, exclusions, 15, 15, 15, 5.0, 138.935, max_multipole=max_multipole)
    pme = CoulombPME(neighbor_list, exclusions, 32, 32, 32, 5, 5.0, 138.935, max_multipole=max_multipole)
    for include_direct, include_reciprocal in [(True,False), (False,True), (True,True)]:
        ewald_energy = ewald(positions, charges, box_vectors, include_direct, include_reciprocal, dipoles)
        pme_energy = pme(positions, charges, box_vectors, include_direct, include_reciprocal, dipoles)
        assert torch.allclose(ewald_energy, pme_energy, rtol=1e-3)
        ewald_energy.backward()
        ewald_grad = positions.grad.clone()
        positions.grad.zero_()
        pme_energy.backward()
        pme_grad = positions.grad.clone()
        positions.grad.zero_()
        assert torch.allclose(ewald_grad, pme_grad, rtol=1e-2, atol=0.1)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_charge_deriv(device):
    """Test derivatives with respect to charge."""
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
    excl = [[0, 6],
            [3, 6]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, requires_grad=True, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, exclusions, 4, 4, 6, 5.25652, 138.935)

    # Compute derivatives of the energies with respect to charges.

    edir = ewald(positions, charges, box_vectors, True, False)
    erecip = ewald(positions, charges, box_vectors, False, True)
    edir.backward(retain_graph=True)
    ddir = charges.grad.clone()
    charges.grad.zero_()
    erecip.backward(retain_graph=True)
    drecip = charges.grad.clone()

    # Compute finite difference approximations from two displaced inputs.

    delta = 0.01
    for i in range(len(charges)):
        c1 = charges.clone()
        c1[i] += delta
        edir1 = ewald(positions, c1, box_vectors, True, False)
        erecip1 = ewald(positions, c1, box_vectors, False, True)
        c2 = charges.clone()
        c2[i] -= delta
        edir2 = ewald(positions, c2, box_vectors, True, False)
        erecip2 = ewald(positions, c2, box_vectors, False, True)
        assert torch.allclose(ddir[i], (edir1-edir2)/(2*delta), rtol=1e-3, atol=1e-3)
        assert torch.allclose(drecip[i], (erecip1-erecip2)/(2*delta), rtol=1e-3, atol=1e-3)

    # Make sure the chain rule is applied properly.

    charges.grad.zero_()
    (2.5*edir).backward()
    ddir2 = charges.grad.clone()
    charges.grad.zero_()
    (2.5*erecip).backward()
    drecip2 = charges.grad.clone()
    assert torch.allclose(2.5*ddir, ddir2)
    assert torch.allclose(2.5*drecip, drecip2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dipole_deriv(device):
    """Test derivatives with respect to dipoles."""
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
    excl = [[0, 6],
            [3, 6]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    dipoles = torch.tensor([[-0.04980356, -0.04964022, 0.0629573],
                            [-0.04855029, -0.02640511, 0.01638032],
                            [0.01484668, 0.0209327, -0.01611683],
                            [0.0081436, -0.06337297, 0.08099023],
                            [0.10179879, -0.06040448, -0.03472181],
                            [0.04007862, -0.05791511, 0.08975159],
                            [0.02838723, -0.02750587, -0.04259318],
                            [-0.0566733, -0.07421055, 0.00508225],
                            [0.0467413, -0.04123889, 0.02388442]],
                           dtype=torch.float32, requires_grad=True, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, exclusions, 5, 5, 7, 4.985823141035867, 138.935, max_multipole='dipole')

    # Compute derivatives of the energies with respect to dipoles.

    edir = ewald(positions, charges, box_vectors, True, False, dipoles)
    erecip = ewald(positions, charges, box_vectors, False, True, dipoles)
    edir.backward(retain_graph=True)
    ddir = dipoles.grad.clone()
    dipoles.grad.zero_()
    erecip.backward(retain_graph=True)
    drecip = dipoles.grad.clone()

    # Compute finite difference approximations from two displaced inputs.

    delta = 0.001
    for i in range(len(dipoles)):
        for j in range(3):
            d1 = dipoles.clone()
            d1[i,j] += delta
            edir1 = ewald(positions, charges, box_vectors, True, False, d1)
            erecip1 = ewald(positions, charges, box_vectors, False, True, d1)
            d2 = dipoles.clone()
            d2[i,j] -= delta
            edir2 = ewald(positions, charges, box_vectors, True, False, d2)
            erecip2 = ewald(positions, charges, box_vectors, False, True, d2)
            assert torch.allclose(ddir[i,j], (edir1-edir2)/(2*delta), rtol=1e-3, atol=1e-3)
            assert torch.allclose(drecip[i,j], (erecip1-erecip2)/(2*delta), rtol=5e-3, atol=1e-3)

    # Make sure the chain rule is applied properly.

    dipoles.grad.zero_()
    (2.5*edir).backward()
    ddir2 = dipoles.grad.clone()
    dipoles.grad.zero_()
    (2.5*erecip).backward()
    drecip2 = dipoles.grad.clone()
    assert torch.allclose(2.5*ddir, ddir2)
    assert torch.allclose(2.5*drecip, drecip2, rtol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batch(device):
    """Test Ewald for a batch of systems."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    num_systems = 10
    num_particles = 20*num_systems
    positions = 5.0*torch.rand((num_particles,3), dtype=torch.float32, device=device)-2.0
    positions.requires_grad_()
    charges = 2.0*torch.rand(num_particles, dtype=torch.float32, device=device)-1.0
    batch = torch.arange(num_systems, device=device).expand((20,-1)).T.flatten()
    box_vectors = []
    for i in range(num_systems):
        scale = 0.9+0.2*torch.rand(1, dtype=torch.float32, device=device)
        box_vectors.append(torch.tensor([[2.0, 0.0, 0.0],
                                         [0.1, 1.6, 0.0],
                                         [0.2, 0.1, 1.5]], dtype=torch.float32, device=device)*scale)
    box_vectors = torch.stack(box_vectors)
    cutoff = 0.4
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, None, 5, 5, 7, 5.0, 138.935)
    energy = ewald(positions, charges, box_vectors, batch=batch)
    for i in range(num_systems):
        mask = batch == i
        energy1 = energy[i]
        energy1.backward(retain_graph=True)
        grad1 = positions.grad[mask]
        pos = torch.tensor(positions[mask], device=device, requires_grad=True)
        box = None if box_vectors is None else box_vectors[i]
        energy2 = ewald(pos, charges[mask], box)
        assert torch.allclose(energy1, energy2, rtol=1e-4)
        energy2.backward()
        grad2 = pos.grad
        assert torch.allclose(grad1, grad2, rtol=1e-4)
        positions.grad.zero_()
        pos.grad.zero_()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('include_direct, include_reciprocal', [(True,False), (False,True), (True,True)])
def test_compute_field(device, include_direct, include_reciprocal):
    """Test computing the electric field."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((30, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(30)], dtype=torch.float32, device=device)
    dipoles = 0.05*(torch.rand((30, 3), dtype=torch.float32, device=device)-0.5)
    box_vectors = torch.tensor([[1, 0, 0], [0, 1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    field_positions = 3*torch.rand((10, 3), dtype=torch.float32, device=device)-1
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    ewald = CoulombEwald(neighbor_list, None, 5, 5, 7, 5.0, 138.935, max_multipole='dipole')
    field = ewald.compute_field(field_positions, positions, charges, box_vectors, include_direct, include_reciprocal, dipoles)

    # Compare the field at each position to the force on a particle of charge 1 at the same position.

    for p, f1 in zip(field_positions, field):
        padded_pos = torch.cat([positions, p.unsqueeze(0)])
        padded_pos.requires_grad_(True)
        padded_charges = torch.nn.functional.pad(charges, pad=(0,1), value=1)
        padded_dipoles = torch.nn.functional.pad(dipoles, pad=(0,0,0,1))
        energy = ewald(padded_pos, padded_charges, box_vectors, include_direct, include_reciprocal, padded_dipoles)
        energy.backward()
        f2 = -padded_pos.grad[-1]
        norm1 = torch.linalg.vector_norm(f1)
        norm2 = torch.linalg.vector_norm(f2)
        diffnorm = torch.linalg.vector_norm(f1-f2)/norm1
        assert torch.allclose(norm1, norm2, rtol=5e-3)
        assert diffnorm < 5e-3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_force_derivatives(device):
    """Test computing derivatives of the force with CoulombEwald."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device, requires_grad=True)-1
    exclusions = torch.tensor([[0,1], [2,3]], dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device, requires_grad=True)
    dipoles = 0.05*torch.randn((9, 3), dtype=torch.float32, device=device, requires_grad=True)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(0.5, device=device)
    ewald = CoulombEwald(neighbor_list, exclusions, 5, 5, 7, 5.0, 138.935, max_multipole='dipole')
    energy = ewald(positions, charges, box_vectors, dipoles=dipoles)
    force = -torch.autograd.grad(energy, positions, create_graph=True)[0]

    # Compute some second derivatives.

    force_norm = torch.linalg.norm(force)
    pos_grad = torch.autograd.grad(force_norm, positions, retain_graph=True)[0]
    charge_grad = torch.autograd.grad(force_norm, charges, retain_graph=True)[0]
    dipole_grad = torch.autograd.grad(force_norm, dipoles)[0]

    # Check the charge derivative against a finite difference approximation.

    delta = 0.05
    for i in range(len(charges)):
        c1 = charges.clone()
        c1[i] += delta
        energy1 = ewald(positions, c1, box_vectors, dipoles=dipoles)
        force_norm1 = torch.linalg.norm(torch.autograd.grad(energy1, positions)[0])
        c2 = charges.clone()
        c2[i] -= delta
        energy2 = ewald(positions, c2, box_vectors, dipoles=dipoles)
        force_norm2 = torch.linalg.norm(torch.autograd.grad(energy2, positions)[0])
        assert torch.allclose(charge_grad[i], (force_norm1-force_norm2)/(2*delta), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that CoulombEwald can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(device=device)
    ewald = CoulombEwald(neighbor_list, None, 4, 4, 6, 5.25652, 138.935)

    # Check that torch.compile works correctly.

    compiled = torch.compile(ewald)
    energy1 = ewald(positions, charges, box_vectors)
    energy2 = compiled(positions, charges, box_vectors)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(ewald)
    ewald2 = pickle.loads(pickled)
    energy3 = ewald2(positions, charges, box_vectors)
    assert torch.allclose(energy1, energy3)
