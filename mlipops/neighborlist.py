import torch
from .utils import periodic_displacements, batch_pairwise_displacements
try:
    import triton
    from .neighborlist_triton import find_sort_keys_kernel, find_neighbors_kernel
    has_triton = True
except ImportError:
    has_triton = False


class NeighborList(torch.nn.Module):
    """Identifies neighboring particles that can interact with each other.

    Neighbors are usually defined by a distance cutoff: any pairs that are closer than the cutoff, possibly taking
    periodic boundary conditions into account, can interact and therefore are returned.  It is also possible to
    omit the cutoff, in which case all pairs are returned regardless of distance.

    Two additional options further restrict the result.  `include_self` determines whether a self interaction pair
    (i, i) should be included for each particle.  `include_symmetric` whether the result should include both of the
    symmetric pairs (i, j) and (j, i), or whether only one should be included.

    You can optionally specify a padding value, in which case all pairs that are within the distance cutoff+padding
    are returned.  This allows saving computation be reusing cached results.  If you call the neighbor list again
    and no particle has moved by more than half the padding distance, the previous result can be returned.  The
    disadvantage is that more pairs need to be included in the neighbor list.  This option is generally useful only
    when the cost of finding neighbors is large compared to the cost of computing interactions based on them.

    .. warning::
        Because NeighborList caches its inputs and outputs, you must never modify them in place after calling the
        NeighborList.  For example, if you call it once with a Tensor of positions, modify the Tensor in place to
        contain different positions, and then pass it to the NeighborList again, it will incorrectly think the positions
        have not changed.
    """
    def __init__(self, cutoff: float | None = None, include_self: bool = False, include_symmetric: bool = False,
                 padding: float | None = None, device: str = 'cpu'):
        """Create a NeighborList for identifying neighbors.

        Parameters
        ----------
        cutoff: float | None
            the cutoff distance to use when identifying neighbors.  If None, all pairs are returned regardless
            of distance.
        include_self: bool
            if True, include self interaction pairs of the form (i, i).
        include_symmetric: bool
            if True, include both of the symmetric pairs (i, j) and (j, i) for particles i and j.  If False, only
            one of the two is included.
        padding: float | None
            the padding distance to add to the cutoff, allowing cached results to be reused.  If None, caching
            is disabled.
        device: str
            the PyTorch device to perform calculation on.
        """
        super().__init__()
        self._cutoff = cutoff
        self._include_self = include_self
        self._include_symmetric = include_symmetric
        self._padding = padding
        self.device = device
        self.use_triton = has_triton and torch.device(device).type == 'cuda'
        self._max_neighbors = 10000
        self._prev_pairs = None
        self._prev_num_particles = None
        self._prev_positions = None
        self._prev_box_vectors = None
        self._prev_batch = None

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def include_self(self):
        return self._include_self

    @property
    def include_symmetric(self):
        return self._include_symmetric

    @property
    def padding(self):
        return self._padding

    def forward(self, positions: torch.Tensor, box_vectors: torch.Tensor | None, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the neighbor list.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (particles, 3) containing the cartesian coordinates of each particle
        box_vectors: torch.Tensor | None
            if batch is None, a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If batch is
            not None, a Tensor of shape (systems, 3, 3) containing the box vectors for each system.  If None, periodic
            boundary conditions are not used.
        batch: torch.Tensor | None
            a Tensor of shape (particles,) containing the index of the system each particle belongs to.  This must be
            sorted in ascending order, and every system must contain at least one particle.  If None, the neighbor list
            is computed for a single system instead of a batch of systems.

        Returns
        -------
        a Tensor of shape (pairs, 2).  Each row contains the indices of two particles that can interact.
        """
        if self._cutoff is None and batch is None:
            # Since we're returning all possible pairs, it mostly doesn't change from one call to the next.
            # We can just return the same value every time.  The one thing we need to check for is that the
            # number of particles hasn't changed.

            num_particles = positions.shape[0]
            if self._prev_pairs is None or num_particles != self._prev_num_particles:
                self._prev_pairs = self._compute_dense_pairs(num_particles, 0)
                self._prev_num_particles = num_particles
            self._prev_batch = batch
            return self._prev_pairs

        # See if we have cached neighbors that are still valid.

        if self._prev_pairs is not None and self._prev_batch is batch and ((self._prev_box_vectors is box_vectors) or
                     (self._prev_box_vectors is not None and box_vectors is not None and box_vectors.equal(self._prev_box_vectors))):
            if positions is self._prev_positions:
                # The positions are exactly the same as before.

                return self._prev_pairs
            if self._padding is not None:
                # If no particle has moved more than half the cutoff distance, we can return the cached neighbors.

                delta = positions-self._prev_positions
                distance = torch.linalg.vector_norm(delta, dim=1)
                if torch.max(distance) < self._padding:
                    return self._prev_pairs

        # Compute a new list of pairs.

        cutoff = self._cutoff
        if self._padding is not None:
            cutoff += self._padding
        if batch is None:
            if self.use_triton and positions.shape[0] > 1000:
                result = self._compute_large_single(positions, box_vectors, cutoff)
                if result is None:
                    # Try again with a larger output buffer.

                    result = self._compute_large_single(positions, box_vectors, cutoff)
            else:
                result = self._compute_small_single(positions, box_vectors, cutoff)
        else:
            result = self._compute_small_batch(positions, box_vectors, cutoff, batch)

        # Cache the result for future use.

        self._prev_positions = positions
        self._prev_box_vectors = box_vectors
        self._prev_pairs = result
        self._prev_batch = batch
        return result

    def _compute_dense_pairs(self, num_particles: int | torch.Tensor, start_index: int | torch.Tensor) -> torch.Tensor:
        """Build a list of the full set of all possible pairs between a set of particles.
        """
        i = torch.arange(start_index, num_particles+start_index, device=self.device)
        if self._include_self:
            if self._include_symmetric:
                return torch.cartesian_prod(i, i)
            else:
                return torch.combinations(i, with_replacement=True)
        else:
            if self._include_symmetric:
                pairs = torch.combinations(i)
                return torch.cat([pairs, pairs.flip(1)])
            else:
                return torch.combinations(i)

    def _compute_small_single(self, positions: torch.Tensor, box_vectors: torch.Tensor | None, cutoff: float) -> torch.Tensor:
        """This implements a brute force algorithm to identify neighbors by testing all possible pairs.  It is
        fast for small numbers of particles but scales as O(n^2), making it less suitable for larger systems.
        """
        # Build matrices of deltas and distances.

        delta = periodic_displacements(positions.view((-1,1,3)) - positions, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=2)

        # Create a mask for which pairs to return.

        mask = distance < cutoff
        if not self._include_symmetric:
            if self._include_self:
                mask = mask.triu()
            else:
                mask = mask.triu(1)
        elif not self._include_self:
            mask.fill_diagonal_(False)

        # Build a matrix of indices and return the results.

        n = positions.shape[0]
        i = torch.arange(n, device=self.device)
        indices = torch.cat((i.view((-1, 1, 1)).expand((n, n, 1)), i.view((1, -1, 1)).expand((n, n, 1))), axis=2)
        return indices[mask]

    def _compute_large_single(self, positions: torch.Tensor, box_vectors: torch.Tensor | None, cutoff: float) -> torch.Tensor:
        """This implements a more complex algorithm for identifying neighbors.  On small systems it is slower than
        _compute_small(), but it becomes much faster as the number of particles grows.  It requires Triton, and
        therefore is only used on GPUs.
        """
        # Sort the particles in a way that groups nearby particles together.

        num_particles = positions.shape[0]
        bin_size = 0.2*cutoff
        grid_size = ((positions.max(dim=0)[0]-positions.min(dim=0)[0])/bin_size).ceil().to(torch.int32)+3
        keys = torch.empty((num_particles,), dtype=torch.int32, device=self.device)
        g = lambda meta: (triton.cdiv(num_particles, meta['BLOCK_SIZE']),)
        find_sort_keys_kernel[g](keys, positions, grid_size, bin_size, num_particles, 256)
        order = keys.sort()[1]
        sorted_positions = positions[order]

        # Compute a bounding box for each block of 32 consecutive particles.

        padding = 32-(num_particles%32)
        if padding != 32:
            padded = torch.nn.functional.pad(sorted_positions, pad=(0,0,0,padding))
            padded[-padding:] = sorted_positions[-1]
            sorted_positions = padded
            order = torch.nn.functional.pad(order, pad=(0,padding), value=num_particles)
        block_positions = sorted_positions.reshape((-1, 32, 3))
        block_min = block_positions.min(dim=1)[0]
        block_max = block_positions.max(dim=1)[0]
        block_width = 0.5*(block_max-block_min)
        block_center = 0.5*(block_min+block_max)
        block_particles = order.reshape((-1, 32))

        # Find pairs of blocks that can interact.

        block_delta = periodic_displacements(block_center.view((-1,1,3)) - block_center, box_vectors)
        block_delta = torch.relu(block_delta.abs()-block_width.view((-1,1,3))-block_width)
        block_distance = torch.linalg.vector_norm(block_delta, dim=2)
        n = block_positions.shape[0]
        i = torch.arange(n, device=self.device)
        indices = torch.cat((i.view((-1, 1, 1)).expand((n, n, 1)), i.view((1, -1, 1)).expand((n, n, 1))), axis=2)
        mask = block_distance < cutoff
        if not self._include_symmetric:
            mask = mask.triu()
        block_pairs = indices[mask]

        # For each pair of blocks, find particles that are within the cutoff.

        output = torch.empty((self._max_neighbors, 2), dtype=torch.int32, device=self.device)
        output_counter = torch.zeros((1,), dtype=torch.int32, device=self.device)
        g = lambda meta: (meta['num_block_pairs'],)
        find_neighbors_kernel[g](output, output_counter, block_pairs, block_particles, block_positions, box_vectors,
                                block_pairs.shape[0], output.shape[0], num_particles, cutoff**2, self._include_self, self._include_symmetric)
        if output_counter > self._max_neighbors:
            # Too many neighbors were found to fit in the output tensor.  Increase the output
            # size and try again. 

            self._max_neighbors = int(1.1*output_counter)
            return None
        output = output.narrow(0, 0, output_counter)
        return output

    def _compute_small_batch(self, positions: torch.Tensor, box_vectors: torch.Tensor | None, cutoff: float,
                             batch: torch.Tensor) -> torch.Tensor:
        """This implements a brute force algorithm to identify neighbors by testing all possible pairs within each
        system.  It is fast for small numbers of particles but scales as O(n^2), making it less suitable for larger
        systems.
        """
        # Identify the range of particles for each system.

        index = (batch[:-1] != batch[1:]).nonzero().flatten()
        start_index = torch.nn.functional.pad(index+1, pad=(1,0))
        system_particles = torch.nn.functional.pad(start_index[1:], pad=(0,1), value=batch.shape[0]) - start_index

        # Build a list of all possible pairs.

        num_systems = start_index.shape[0]
        system_pairs = []
        for i in range(num_systems):
            system_pairs.append(self._compute_dense_pairs(system_particles[i], start_index[i]))
        all_pairs = torch.cat(system_pairs)

        # Filter them by distance.

        delta = batch_pairwise_displacements(positions, all_pairs, batch, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        return all_pairs[distance < cutoff]