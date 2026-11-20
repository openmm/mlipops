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
    torch.Tensor:
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
    torch.Tensor:
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
    torch.Tensor:
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
    torch.Tensor:
        a Tensor of shape (n_pairs, 3).  Each row contains the displacement between the corresponding pair of points.
    """
    displacements = positions[pairs[:,1]] - positions[pairs[:,0]]
    if box_vectors is not None:
        displacements = batch_periodic_displacements(displacements, batch[pairs[:,0]], box_vectors)
    return displacements


def get_covalent_radii(atomic_numbers: torch.Tensor, bohr_radius: float) -> torch.Tensor:
    """Get the covalent radii for a set of atoms based on their atomic numbers.

    Covalent radii are not uniquely defined.  Various sets of values have been published.  The ones used here are chosen
    to be consistent with the simple-dftd3 library (https://github.com/dftd3/simple-dftd3).  They are taken from
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


def create_supercell(positions: torch.Tensor, box_vectors: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tile copies of a unit cell to create a larger supercell whose width is at least twice the cutoff distance in
    all dimensions.

    Parameters
    ----------
    positions: torch.Tensor
        a Tensor of shape (n_points, 3) containing the Cartesian coordinates of each point in the unit cell
    box_vectors: torch.Tensor
        a Tensor of shape (3, 3) containing box vectors defining the unit cell dimensions
    cutoff: float
        the cutoff distance

    Returns
    -------
    torch.Tensor:
        a Tensor of shape (n_new_points, 3) containing the Cartesian coordinates of each point in the supercell
    torch.Tensor:
        a Tensor of shape (3, 3) containing box vectors defining the supercell dimensions
    torch.Tensor:
        a Tensor of shape (n_new_points,).  For each point in the supercell, it contains the index of the point in the
        unit cell to which it corresponds.  For example, if the unit cell contains three points and the supercell
        contains two unit cells, this would be [0, 1, 2, 0, 1, 2].
    torch.Tensor:
        a Tensor of shape (n_new_points, 3) containing the offset that was applied to each point in the supercell,
        measured as an integer number of unit cell box vectors
    """
    num_particles = positions.shape[0]
    copies = (2*cutoff/box_vectors.diag()).ceil().to(torch.int32)
    total_copies = copies.prod()
    new_box_vectors = copies.unsqueeze(1)*box_vectors
    particle = torch.arange(num_particles).tile((total_copies,))
    cell = torch.arange(total_copies).repeat(num_particles, 1).T.flatten()
    cell_offset = torch.cartesian_prod(torch.arange(copies[0]), torch.arange(copies[1]), torch.arange(copies[2])).to(positions.dtype)
    new_positions = positions.tile((total_copies, 1))+(cell_offset@box_vectors)[cell]
    return new_positions, new_box_vectors, particle, cell_offset[cell]


def batch_create_supercell(positions: torch.Tensor, box_vectors: torch.Tensor, cutoff: float, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create supercells for a batch of systems by tiling copies of each one's unit cell.

    This function is like create_supercell(), but it operates on multiple systems at once.  Each system is tiled to
    ensure the periodic box width is at least twice the cutoff distance in all dimensions.  Because each system can have
    different box vectors, the number of unit cells in the supercell can be different for each one.

    Parameters
    ----------
    positions: torch.Tensor
        a Tensor of shape (n_points, 3) containing the Cartesian coordinates of each point in the unit cell
    box_vectors: torch.Tensor | None
        a Tensor of shape (n_systems, 3, 3) containing box vectors defining the periodic box for each system
    cutoff: float
        the cutoff distance
    batch: torch.Tensor
        a Tensor of shape (n_points,) containing the index of the system each point belongs to

    Returns
    -------
    torch.Tensor:
        a Tensor of shape (n_new_points, 3) containing the Cartesian coordinates of each point in the supercell
    torch.Tensor:
        a Tensor of shape (n_systems, 3, 3) containing box vectors defining the supercell dimensions for each system
    torch.Tensor:
        a Tensor of shape (n_new_points,).  For each point in the supercell, it contains the index of the point in the
        unit cell to which it corresponds.  This should be interpreted as an index into the input positions Tensor.
        That is, it refers to the global list of all points in all systems.
    torch.Tensor:
        a Tensor of shape (n_new_points, 3) containing the offset that was applied to each point in the supercell,
        measured as an integer number of unit cell box vectors
    torch.Tensor:
        a Tensor of shape (n_new_points,) containing the index of the system each point in the supercell belongs to
    """
    split_locations = torch.nonzero(batch[1:] != batch[:-1]).flatten()+1
    batch_positions = torch.tensor_split(positions, split_locations)
    new_positions = []
    new_box_vectors = []
    particle = []
    offset = []
    new_batch = []
    start_index = 0
    for i in range(len(batch_positions)):
        batch_new_positions, batch_new_box_vectors, batch_particle, batch_offset = create_supercell(batch_positions[i], box_vectors[i], cutoff)
        new_positions.append(batch_new_positions)
        new_box_vectors.append(batch_new_box_vectors)
        particle.append(batch_particle+start_index)
        offset.append(batch_offset)
        new_batch.append(i*torch.ones(batch_new_positions.size()[0], dtype=torch.int32, device=batch.device))
        start_index += batch_positions[i].shape[0]
    return torch.cat(new_positions), torch.stack(new_box_vectors), torch.cat(particle), torch.cat(offset), torch.cat(new_batch)


def compute_supercell_neighbors(neighbor_list: "NeighborList", positions: torch.Tensor,
                                unitcell_box_vectors: torch.Tensor, supercell_box_vectors: torch.Tensor,
                                points: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Compute a neighbor list for a unit cell within a supercell created by create_supercell().

    If the supercell contains N unit cells, the neighbor list returned by this function has N times fewer pairs than the
    full neighbor list for all points in the supercell.  It excludes the redundant pairs created by applying identical
    translations to the same points.

    Another way of thinking about this is that if you compute the energy of a pairwise interaction for the returned
    pairs, it will give the energy of a single unit cell, which is 1/N of the energy of the supercell.

    Parameters
    ----------
    neighbor_list: NeighborList
        the NeighborList that will be used to indentify pairs of points within the cutoff distance
    positions: torch.Tensor
        the Tensor of Cartesian coordinates that was returned by create_supercell().  Its shape is (num_points, 3).
    unitcell_box_vectors: torch.Tensor
        the unit cell box vectors that were passed to create_supercell().  Its shape is (3, 3).
    supercell_box_vectors: torch.Tensor
        the supercell box vectors returned by create_supercell().  Its shape is (3, 3).
    points: torch.Tensor
        the Tensor of point indices returned by create_supercell().  Its shape is (num_points,).
    offsets: torch.Tensor
        the Tensor of offsets returned by create_supercell().  Its shape is (num_points, 3).

    Returns
    -------
    torch.Tensor:
        a Tensor of shape (pairs, 2).  Each row contains the indices of two particles that can interact.
    """
    in_first_cell = torch.all(offsets == 0, dim=1)
    subset = torch.arange(offsets.shape[0])[in_first_cell]
    pairs = neighbor_list(positions, supercell_box_vectors, subset=subset)
    if not neighbor_list.include_symmetric:
        p0 = pairs[:,0]
        p1 = pairs[:,1]
        n0 = points[p0]
        n1 = points[p1]
        copies = torch.round(supercell_box_vectors.diag()/unitcell_box_vectors.diag())
        pair_offset = offsets[p1]-offsets[p0]
        pair_offset -= torch.round(pair_offset/copies)*copies
        offset_dir = pair_offset.sign()
        positive = (offset_dir[:,0]*copies[1] + offset_dir[:,1])*copies[2] + offset_dir[:,2] > 0
        mask = (n0<n1) | ((n0==n1) & positive)
        if neighbor_list.include_self:
            mask |= p0 == p1
        pairs = pairs[mask]
    return pairs


def batch_compute_supercell_neighbors(neighbor_list: "NeighborList", positions: torch.Tensor,
                                      unitcell_box_vectors: torch.Tensor, supercell_box_vectors: torch.Tensor,
                                      points: torch.Tensor, offsets: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Compute a neighbor list for unit cells in a batch of supercells created by batch_create_supercell().

    This function is similar to compute_supercell_neighbors(), but it operates on multiple systems at once.

    Parameters
    ----------
    neighbor_list: NeighborList
        the NeighborList that will be used to indentify pairs of points within the cutoff distance
    positions: torch.Tensor
        the Tensor of Cartesian coordinates that was returned by batch_create_supercell().  Its shape is (num_points, 3).
    unitcell_box_vectors: torch.Tensor
        the unit cell box vectors that were passed to batch_create_supercell().  Its shape is (n_systems, 3, 3).
    supercell_box_vectors: torch.Tensor
        the supercell box vectors returned by batch_create_supercell().  Its shape is (n_systems, 3, 3).
    points: torch.Tensor
        the Tensor of point indices returned by batch_create_supercell().  Its shape is (num_points,).
    offsets: torch.Tensor
        the Tensor of offsets returned by batch_create_supercell().  Its shape is (num_points, 3).
    batch: torch.Tensor
        a Tensor of shape (n_points,) containing the index of the system each point belongs to

    Returns
    -------
    torch.Tensor:
        a Tensor of shape (pairs, 2).  Each row contains the indices of two particles that can interact.
    """
    in_first_cell = torch.all(offsets == 0, dim=1)
    subset = torch.arange(offsets.shape[0])[in_first_cell]
    pairs = neighbor_list(positions, supercell_box_vectors, batch=batch, subset=subset)
    if not neighbor_list.include_symmetric:
        p0 = pairs[:,0]
        p1 = pairs[:,1]
        n0 = points[p0]
        n1 = points[p1]
        pair_batch = batch[p0]
        copies = torch.round(supercell_box_vectors.diagonal(dim1=1, dim2=2)/unitcell_box_vectors.diagonal(dim1=1, dim2=2))[pair_batch]
        pair_offset = offsets[p1]-offsets[p0]
        pair_offset -= torch.round(pair_offset/copies)*copies
        offset_dir = pair_offset.sign()
        positive = (offset_dir[:,0]*copies[:,1] + offset_dir[:,1])*copies[:,2] + offset_dir[:,2] > 0
        mask = (n0<n1) | ((n0==n1) & positive)
        if neighbor_list.include_self:
            mask |= p0 == p1
        pairs = pairs[mask]
    return pairs
