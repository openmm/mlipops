import torch
try:
    import triton
    from .utils_triton import batch_periodic_displacements_kernel
    has_triton = True
except ImportError:
    has_triton = False


def periodic_displacements(displacements: torch.Tensor, box_vectors: torch.Tensor | None) -> torch.Tensor:
    """Apply periodic boundary conditions to a 1D to 2D set of displacement vectors.

    Parameters
    ----------
    displacements: torch.Tensor
        a Tensor of shape (M, 3) or (M, N, 3) where M and N are arbitrary.  Each of the M or M*N displacement vectors
        should be the difference between a pair of positions in 3D space.
    box_vectors: torch.Tensor | None
        a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
        conditions are not used and the displacement vectors are returned unchanged.

    Returns
    -------
    a Tensor of the same shape as displacements.  The vectors have been modified to apply periodic boundary conditions.
    """
    if box_vectors is not None:
        if displacements.ndim == 2:
            scale = torch.round(displacements[:,2]/box_vectors[2,2])
            displacements = displacements - scale.unsqueeze(1)*box_vectors[2].view((1,3)).expand((-1,3))
            scale = torch.round(displacements[:,1]/box_vectors[1,1])
            displacements = displacements - scale.unsqueeze(1)*box_vectors[1].view((1,3)).expand((-1,3))
            scale = torch.round(displacements[:,0]/box_vectors[0,0])
            displacements = displacements - scale.unsqueeze(1)*box_vectors[0].view((1,3)).expand((-1,3))
        else:
            scale = torch.round(displacements[:,:,2]/box_vectors[2,2])
            displacements = displacements - scale.unsqueeze(2)*box_vectors[2].view((1,1,3)).expand((-1,-1,3))
            scale = torch.round(displacements[:,:,1]/box_vectors[1,1])
            displacements = displacements - scale.unsqueeze(2)*box_vectors[1].view((1,1,3)).expand((-1,-1,3))
            scale = torch.round(displacements[:,:,0]/box_vectors[0,0])
            displacements = displacements - scale.unsqueeze(2)*box_vectors[0].view((1,1,3)).expand((-1,-1,3))
    return displacements


def batch_periodic_displacements(displacements: torch.Tensor, batch: torch.Tensor, box_vectors: torch.Tensor | None) -> torch.Tensor:
    """Apply periodic boundary conditions to displacement vectors for a batch of systems.

    Parameters
    ----------
    displacements: torch.Tensor
        a Tensor of shape (M, 3) where M is arbitrary.  Each displacement vector should be the difference between a pair
        of positions in 3D space.
    batch: torch.Tensor | None
        a Tensor of shape (M,) containing the index of the system each vector belongs to
    box_vectors: torch.Tensor | None
        a Tensor of shape (n_systems, 3, 3) containing box vectors defining the periodic box for each system.  If None,
        periodic boundary conditions are not used and the displacement vectors are returned unchanged.

    Returns
    -------
    a Tensor of the same shape as displacements.  The vectors have been modified to apply periodic boundary conditions.
    """
    if box_vectors is not None:
        if has_triton and displacements.device.type == 'cuda':
            g = lambda meta: (triton.cdiv(batch.shape[0], meta['BLOCK_SIZE']),)
            batch_periodic_displacements_kernel[g](displacements, batch, box_vectors, batch.shape[0], 256)
        else:
            box = box_vectors[batch]
            scale = torch.round(displacements[:,2]/box[:,2,2])
            displacements = displacements - scale.unsqueeze(1)*box[:,2]
            scale = torch.round(displacements[:,1]/box[:,1,1])
            displacements = displacements - scale.unsqueeze(1)*box[:,1]
            scale = torch.round(displacements[:,0]/box[:,0,0])
            displacements = displacements - scale.unsqueeze(1)*box[:,0]
    return displacements


def pairwise_displacements(positions: torch.Tensor, pairs: torch.Tensor, box_vectors: torch.Tensor | None) -> torch.Tensor:
    """Compute the displacement between pairs of points, optionally taking periodic boundary conditions into
    account.

    Parameters
    ----------
    positions: torch.Tensor
        a Tensor of shape (n_points, 3) containing the Cartesian coordinates of each point
    pairs: torch.Tensor
        a Tensor of shape (n_pairs, 2).  Each row contains the indices of two points whose displacement should
        be computed.
    box_vectors: torch.Tensor | None
        a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
        conditions are not used.

    Returns
    -------
    a Tensor of shape (n_pairs, 3).  Each row contains the displacement between the corresponding pair of points.
    """
    return periodic_displacements(positions[pairs[:,1]] - positions[pairs[:,0]], box_vectors)


def batch_pairwise_displacements(positions: torch.Tensor, pairs: torch.Tensor, batch: torch.Tensor,
                                 box_vectors: torch.Tensor | None) -> torch.Tensor:
    """Compute the displacement between pairs of points in a batch of systems, optionally taking periodic boundary
    conditions into account.

    Parameters
    ----------
    positions: torch.Tensor
        a Tensor of shape (n_points, 3) containing the Cartesian coordinates of each point
    pairs: torch.Tensor
        a Tensor of shape (n_pairs, 2).  Each row contains the indices of two points whose displacement should
        be computed.
    batch: torch.Tensor | None
        a Tensor of shape (n_points,) containing the index of the system each point belongs to
    box_vectors: torch.Tensor | None
        a Tensor of shape (n_systems, 3, 3) containing box vectors defining the periodic box for each system.  If None,
        periodic boundary conditions are not used.

    Returns
    -------
    a Tensor of shape (n_pairs, 3).  Each row contains the displacement between the corresponding pair of points.
    """
    displacements = positions[pairs[:,1]] - positions[pairs[:,0]]
    if box_vectors is not None:
        displacements = batch_periodic_displacements(displacements, batch[pairs[:,0]], box_vectors)
    return displacements


def get_covalent_radii(atomic_numbers: torch.Tensor, bohr_radius: float) -> torch.Tensor:
    """Get the covalent radii for a set of atoms based on their atomic numbers.

    Covalent radii are not uniquely defined.  Various sets of values have been published.  The ones used here are chosen
    to be consistent with the simple-dfdt3 library (https://github.com/dftd3/simple-dftd3).  They are taken from
    Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197, except that the radii of metals have been reduced by 10%.

    Parameters
    ----------
    atomic_numbers: torch.Tensor
        a Tensor of shape (n_atoms,) containing the atomic number of each atom
    bohr_radius: float
        the Bohr radius.  This sets the units.  Pass 1.0 to get radii in Bohr, 0.052917721 to get radii in nanometers,
        or 0.52917721 to get radii in Angstroms.
    """
    radii = [0, 32, 46, 120, 94, 77, 75, 71, 63, 64, 67, 140, 125, 112, 104, 110, 102, 99, 96, 176, 154, 133, 122,
             121, 110, 107, 104, 100, 99, 101, 109, 112, 109, 114, 110, 113, 117, 189, 167, 147, 139, 132, 124, 114,
             112, 112, 108, 114, 123, 128, 126, 126, 123, 132, 131, 209, 176, 162, 147, 158, 157, 156, 155, 151,
             152, 151, 150, 149, 149, 148, 153, 146, 137, 131, 123, 118, 115, 111, 112, 112, 132, 130, 130, 136,
             131, 138, 142, 200, 181, 167, 158, 152, 153, 154, 155, 149, 149, 151, 151, 148, 150, 156, 158, 145,
             141, 134, 129, 127, 121, 115, 114, 109, 122, 136, 143, 146, 158, 148, 157]
    covalent_radii = torch.tensor(radii, dtype=torch.float32, device=atomic_numbers.device)*bohr_radius/52.9177210903
    return covalent_radii[atomic_numbers]
