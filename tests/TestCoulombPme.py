import torch
import pickle
import pytest
from mlipops import NeighborList, CoulombPME


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_rectangular(device):
    """Test PME on a rectangular box."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    pme = CoulombPME(neighbor_list, None, 14, 15, 16, 5, 4.985823141035867, 138.935, cutoff)
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

    edirect = pme(positions, charges, box_vectors, True, False)
    assert torch.allclose(torch.tensor(0.5811535194516182), edirect)
    erecip = pme(positions, charges, box_vectors, False, True)
    assert torch.allclose(torch.tensor(-90.92361028496651), erecip)
    etotal = pme(positions, charges, box_vectors)
    assert torch.allclose(etotal, edirect+erecip)
    expected_ddirect = [[-0.4068958163, 1.128490567, 0.2531163692],
                        [8.175477028, -15.20702648, -5.499810219],
                        [-0.2548360825, 0.003096142784, -0.67370224],
                        [0.09854402393, 0.5804504156, 1.063418627],
                        [-0, -0, -0],
                        [-7.859698296, 14.16478539, 5.236941814],
                        [0.684042871, -1.312145352, 0.7057141662],
                        [30.47141075, 6.726415634, -6.697656631],
                        [-30.90804291, -6.084065914, 5.611977577]]
    expected_drecip = [[-0.6407046318, -27.59628105, -3.745499372],
                       [30.76446915, -27.10591507, -82.14082336],
                       [-15.06353951, 10.37030602, -38.38755035],
                       [-7.421859741, 21.9861393, 39.86354828],
                       [-0, -0, -0],
                       [-13.09759808, 6.393665314, 34.15939713],
                       [19.53832817, -59.55260849, 33.96843338],
                       [122.5542908, 60.35510254, -27.44270515],
                       [-136.679245, 15.14429855, 43.89074326]]
    edirect.backward()
    assert torch.allclose(torch.tensor(expected_ddirect), positions.grad.cpu(), rtol=1e-4, atol=1e-5)
    positions.grad.zero_()
    erecip.backward()
    assert torch.allclose(torch.tensor(expected_drecip), positions.grad.cpu(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_triclinic(device):
    """Test PME on a triclinic box."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    pme = CoulombPME(neighbor_list, None, 14, 16, 15, 5, 5.0, 138.935, cutoff)
    pos = [[1.31396193, -0.9377441519, 0.9009447048],
           [1.246411648, 0.4955210369, -0.3256100634],
           [-0.4058114057, 1.281592137, -0.4926674903],
           [-0.7349805575, 1.056079455, 1.860180039],
           [-0.988155201, 0.5365767902, 1.437862885],
           [0.8375782005, 1.165265952, -0.1243717955],
           [1.753322368, 1.14372735, 0.627633104],
           [-0.5734898572, 0.1200222802, 1.022400845],
           [0.3254995233, 0.30204198, 0.8533009354]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [-0.1, 1.2, 0], [0.2, -0.15, 1.1]], dtype=torch.float32, device=device)

    # Compare forces and energies to values computed with OpenMM.

    edirect = pme(positions, charges, box_vectors, True, False)
    assert torch.allclose(torch.tensor(-178.86083489656448), edirect)
    erecip = pme(positions, charges, box_vectors, False, True)
    assert torch.allclose(torch.tensor(-200.9420623172533), erecip)
    expected_ddirect = [[-1000.97644, -326.2085571, 373.3143005],
                        [401.765686, 153.7181702, -278.0073242],
                        [2140.490723, -633.4395752, -1059.523071],
                        [-1.647740602, 10.02025795, 0.2182842493],
                        [-0, -0, -0],
                        [0.05209997296, -2.530653, 3.196420431],
                        [-2139.176758, 633.9973145, 1060.562622],
                        [13.49786377, 11.52490139, -10.12783146],
                        [585.994812, 152.9181519, -89.63345337]]
    expected_drecip = [[-162.9051514, 32.17734528, -77.43495178],
                       [11.11517906, 52.98329163, -83.18161011],
                       [34.50453186, 8.428194046, -4.691772938],
                       [-12.71308613, 20.7514267, -13.68377304],
                       [-0, -0, -0],
                       [8.277475357, -3.927520275, 13.88403988],
                       [-34.93006897, -7.739934444, 8.986465454],
                       [45.33776474, -36.9358139, 40.34444809],
                       [111.2698975, -65.63329315, 115.8478012]]
    edirect.backward()
    assert torch.allclose(torch.tensor(expected_ddirect), positions.grad.cpu(), rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert torch.allclose(torch.tensor(expected_drecip), positions.grad.cpu(), rtol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exclusions(device):
    """Test PME with exclusions."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    pos = [[1.31396193, -0.9377441519, 0.9009447048],
           [1.246411648, 0.4955210369, -0.3256100634],
           [-0.4058114057, 1.281592137, -0.4926674903],
           [-0.7349805575, 1.056079455, 1.860180039],
           [-0.988155201, 0.5365767902, 1.437862885],
           [0.8375782005, 1.165265952, -0.1243717955],
           [1.753322368, 1.14372735, 0.627633104],
           [-0.5734898572, 0.1200222802, 1.022400845],
           [0.3254995233, 0.30204198, 0.8533009354]]
    excl = [[0, 3],
            [2, 3],
            [8, 7]]
    positions = torch.tensor(pos, dtype=torch.float32, requires_grad=True, device=device)
    exclusions = torch.tensor(excl, dtype=torch.int32, device=device)
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [-0.1, 1.2, 0], [0.2, -0.15, 1.1]], dtype=torch.float32, device=device)
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    pme = CoulombPME(neighbor_list, exclusions, 14, 16, 15, 5, 5.0, 138.935, cutoff)

    # Compare forces and energies to values computed with OpenMM.

    edirect = pme(positions, charges, box_vectors, True, False)
    assert torch.allclose(torch.tensor(-204.22671127319336), edirect)
    erecip = pme(positions, charges, box_vectors, False, True)
    assert torch.allclose(torch.tensor(-200.9420623172533), erecip)
    expected_ddirect = [[-998.2406773, -314.4639407, 379.7956738],
                        [401.7656421, 153.7181283, -278.0072042],
                        [2136.789297, -634.4331203, -1062.13192],
                        [-0.6838558404, -0.7345126528, -3.655667043],
                        [-0, -0, -0],
                        [0.05210044985, -2.530651058, 3.196419874],
                        [-2139.175743, 634.0007806, 1060.564263],
                        [21.9532636, -40.74009123, 38.42738517],
                        [577.5399728, 205.183407, -138.1889512]]
    expected_drecip = [[-162.9051514, 32.17734528, -77.43495178],
                       [11.11517906, 52.98329163, -83.18161011],
                       [34.50453186, 8.428194046, -4.691772938],
                       [-12.71308613, 20.7514267, -13.68377304],
                       [-0, -0, -0],
                       [8.277475357, -3.927520275, 13.88403988],
                       [-34.93006897, -7.739934444, 8.986465454],
                       [45.33776474, -36.9358139, 40.34444809],
                       [111.2698975, -65.63329315, 115.8478012]]
    edirect.backward()
    assert torch.allclose(torch.tensor(expected_ddirect), positions.grad.cpu(), rtol=1e-4)
    positions.grad.zero_()
    erecip.backward()
    assert torch.allclose(torch.tensor(expected_drecip), positions.grad.cpu(), rtol=1e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_dipoles(device):
    """Test Coulomb with dipoles."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    exclusions = torch.tensor([[1,6]], dtype=torch.int32, device=device)
    pme = CoulombPME(neighbor_list, exclusions, 28, 32, 30, 5, 5.0, 138.935, cutoff, 'dipole')
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
    dipoles = torch.tensor([[-0.04980356, -0.04964022, 0.0629573],
                            [-0.04855029, -0.02640511, 0.01638032],
                            [0.01484668, 0.0209327, -0.01611683],
                            [0.0081436, -0.06337297, 0.08099023],
                            [0.10179879, -0.06040448, -0.03472181],
                            [0.04007862, -0.05791511, 0.08975159],
                            [0.02838723, -0.02750587, -0.04259318],
                            [-0.0566733, -0.07421055, 0.00508225],
                            [0.0467413, -0.04123889, 0.02388442]], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [-0.1, 1.2, 0], [0.2, -0.15, 1.1]], dtype=torch.float32, device=device)

    # Compare forces and energies to values computed with OpenMM.

    energy = pme(positions, charges, box_vectors, dipoles=dipoles)
    assert torch.allclose(torch.tensor(-51.28904168314841), energy, rtol=1e-3)
    expected_forces = [[113.81919760108624, 296.6104441965564, 128.9728353186539],
                       [-214.14110749628404, -127.04697783809274, 125.62562742356016],
                       [-37.22094717568561, -33.78632972234377, 42.20070828269675],
                       [254.5522531801907, 420.4298696380457, -2.1721181630735344],
                       [448.8463639257915, 548.8092079882196, 94.70003305947873],
                       [123.64106541156079, 207.57091036526973, -244.07758199528342],
                       [104.63619639029427, 182.04026660608343, -115.13324445618028],
                       [-1018.5373205705725, -1498.5142184607134, -50.31094106504052],
                       [224.33700096205, 3.428229393838457, 20.043140796737404]]
    energy.backward()
    assert torch.allclose(torch.tensor(expected_forces), -positions.grad.cpu(), rtol=5e-3, atol=1e-3)


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
    pme = CoulombPME(neighbor_list, exclusions, 14, 15, 16, 5, 4.985823141035867, 138.935)

    # Compute derivatives of the energies with respect to charges.

    edir = pme(positions, charges, box_vectors, True, False)
    erecip = pme(positions, charges, box_vectors, False, True)
    edir.backward(retain_graph=True)
    ddir = charges.grad.clone()
    charges.grad.zero_()
    erecip.backward(retain_graph=True)
    drecip = charges.grad.clone()

    # Compute finite difference approximations from two displaced inputs.

    delta = 0.001
    for i in range(len(charges)):
        c1 = charges.clone()
        c1[i] += delta
        edir1 = pme(positions, c1, box_vectors, True, False)
        erecip1 = pme(positions, c1, box_vectors, False, True)
        c2 = charges.clone()
        c2[i] -= delta
        edir2 = pme(positions, c2, box_vectors, True, False)
        erecip2 = pme(positions, c2, box_vectors, False, True)
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
    pme = CoulombPME(neighbor_list, exclusions, 14, 15, 16, 5, 4.985823141035867, 138.935, max_multipole='dipole')

    # Compute derivatives of the energies with respect to dipoles.

    edir = pme(positions, charges, box_vectors, True, False, dipoles)
    erecip = pme(positions, charges, box_vectors, False, True, dipoles)
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
            edir1 = pme(positions, charges, box_vectors, True, False, d1)
            erecip1 = pme(positions, charges, box_vectors, False, True, d1)
            d2 = dipoles.clone()
            d2[i,j] -= delta
            edir2 = pme(positions, charges, box_vectors, True, False, d2)
            erecip2 = pme(positions, charges, box_vectors, False, True, d2)
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
    assert torch.allclose(2.5*drecip, drecip2)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_double_derivative(device):
    """Test that asking for a second derivative throws an exception."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    positions.requires_grad_()
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    charges.requires_grad_()
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(device=device)
    pme = CoulombPME(neighbor_list, None, 14, 16, 15, 5, 5.0, 138.935)
    edir = pme(positions, charges, box_vectors, True, False)
    erecip = pme(positions, charges, box_vectors, False, True)
    ddir = torch.autograd.grad(edir, positions, retain_graph=True)
    drecip = torch.autograd.grad(erecip, positions, retain_graph=True)
    with pytest.raises(Exception):
        torch.autograd.grad(ddir, positions, retain_graph=True)
    with pytest.raises(Exception):
        torch.autograd.grad(drecip, positions, retain_graph=True)
    with pytest.raises(Exception):
        torch.autograd.grad(ddir, charges, retain_graph=True)
    with pytest.raises(Exception):
        torch.autograd.grad(drecip, charges, retain_graph=True)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batch(device):
    """Test PME for a batch of systems."""
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
    pme = CoulombPME(neighbor_list, None, 14, 16, 15, 5, 5.0, 138.935)
    energy = pme(positions, charges, box_vectors, batch=batch)
    for i in range(num_systems):
        mask = batch == i
        energy1 = energy[i]
        energy1.backward(retain_graph=True)
        grad1 = positions.grad[mask]
        pos = torch.tensor(positions[mask], device=device, requires_grad=True)
        box = None if box_vectors is None else box_vectors[i]
        energy2 = pme(pos, charges[mask], box)
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
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    field_positions = 3*torch.rand((10, 3), dtype=torch.float32, device=device)-1
    cutoff = 0.5
    neighbor_list = NeighborList(cutoff, device=device)
    pme = CoulombPME(neighbor_list, None, 14, 16, 15, 5, 5.0, 138.935, max_multipole='dipole')
    field = pme.compute_field(field_positions, positions, charges, box_vectors, include_direct, include_reciprocal, dipoles)

    # Compare the field at each position to the force on a particle of charge 1 at the same position.

    for p, f1 in zip(field_positions, field):
        padded_pos = torch.cat([positions, p.unsqueeze(0)])
        padded_pos.requires_grad_(True)
        padded_charges = torch.nn.functional.pad(charges, pad=(0,1), value=1)
        padded_dipoles = torch.nn.functional.pad(dipoles, pad=(0,0,0,1))
        energy = pme(padded_pos, padded_charges, box_vectors, include_direct, include_reciprocal, padded_dipoles)
        energy.backward()
        f2 = -padded_pos.grad[-1]
        norm1 = torch.linalg.vector_norm(f1)
        norm2 = torch.linalg.vector_norm(f2)
        diffnorm = torch.linalg.vector_norm(f1-f2)/norm1
        assert torch.allclose(norm1, norm2, rtol=5e-3)
        assert diffnorm < 5e-3


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compile_and_pickle(device):
    """Test that CoulombPME can be compiled and pickled."""
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('No GPU')
    positions = 3*torch.rand((9, 3), dtype=torch.float32, device=device)-1
    charges = torch.tensor([(i-4)*0.1 for i in range(9)], dtype=torch.float32, device=device)
    box_vectors = torch.tensor([[1, 0, 0], [0,1.1, 0], [0, 0, 1.2]], dtype=torch.float32, device=device)
    neighbor_list = NeighborList(device=device)
    pme = CoulombPME(neighbor_list, None, 14, 16, 15, 5, 5.0, 138.935)

    # Check that torch.compile works correctly.

    compiled = torch.compile(pme)
    energy1 = pme(positions, charges, box_vectors)
    energy2 = compiled(positions, charges, box_vectors)
    assert torch.allclose(energy1, energy2)

    # Check that pickle works correctly.

    pickled = pickle.dumps(pme)
    pme2 = pickle.loads(pickled)
    energy3 = pme2(positions, charges, box_vectors)
    assert torch.allclose(energy1, energy3)
