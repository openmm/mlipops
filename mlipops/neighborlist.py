import torch
try:
    import triton
    from neighborlist_triton import find_sort_keys_kernel, find_neighbors_kernel
    has_triton = True
except ImportError:
    has_triton = False


class NeighborList(torch.nn.Module):
    def __init__(self, cutoff: float | None = None, include_self: bool = False, include_symmetric: bool = False,
                 padding: float | None = None, device: str = 'cpu'):
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

    def forward(self, positions: torch.Tensor, box_vectors: torch.Tensor | None):
        if self._cutoff is None:
            # Since we're returning all possible pairs, it mostly doesn't change from one call to the next.
            # We can just return the same value every time.  The one thing we need to check for is that the
            # number of particles hasn't changed.

            num_particles = positions.shape[0]
            if self._prev_pairs is None or num_particles != self._prev_num_particles:
                i = torch.arange(num_particles, device=positions.device)
                if self._include_self:
                    if self._include_symmetric:
                        self._prev_pairs = torch.cartesian_prod(i, i)
                    else:
                        self._prev_pairs = torch.combinations(i, with_replacement=True)
                else:
                    if self._include_symmetric:
                        pairs = torch.combinations(i)
                        self._prev_pairs = torch.cat([pairs, pairs.flip(1)])
                    else:
                        self._prev_pairs = torch.combinations(i)
                self._prev_num_particles = num_particles
            return self._prev_pairs

        # If no particle has moved more than half the cutoff distance, we can return the cached neighbors.

        if self._padding is not None and self._prev_pairs is not None:
            delta = positions-self._prev_positions
            distance = torch.linalg.vector_norm(delta, dim=1)
            if torch.max(distance) < self._padding:
                return self._prev_pairs

        # Compute a new list of pairs.

        cutoff = self._cutoff
        if self._padding is not None:
            cutoff += self._padding
        if self.use_triton and positions.shape[0] > 1000:
            result = self.compute_large(positions, box_vectors, cutoff)
            if result is None:
                # Try again with a larger output buffer.

                result = self.compute_large(positions, box_vectors, cutoff)
        result = self.compute_small(positions, box_vectors, cutoff)

        # Cache the result for future use.

        if self._padding is not None:
            self._prev_positions = positions
            self._prev_pairs = result
        return result

    def compute_small(self, positions: torch.Tensor, box_vectors: torch.Tensor | None, cutoff: float):
        # Build matrices of deltas and distances.

        delta = positions.view((-1,1,3)) - positions
        if box_vectors is not None:
            scale = torch.round(delta[:,:,2]/box_vectors[2,2])
            delta = delta - scale.unsqueeze(2)*box_vectors[2].view((1,1,3)).expand((-1,-1,3))
            scale = torch.round(delta[:,:,1]/box_vectors[1,1])
            delta = delta - scale.unsqueeze(2)*box_vectors[1].view((1,1,3)).expand((-1,-1,3))
            scale = torch.round(delta[:,:,0]/box_vectors[0,0])
            delta = delta - scale.unsqueeze(2)*box_vectors[0].view((1,1,3)).expand((-1,-1,3))
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

    def compute_large(self, positions: torch.Tensor, box_vectors: torch.Tensor | None, cutoff: float):
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

        block_delta = block_center.view((-1,1,3)) - block_center
        if box_vectors is not None:
            scale = torch.round(block_delta[:,:,2]/box_vectors[2,2])
            block_delta = block_delta - scale.unsqueeze(2)*box_vectors[2].view((1,1,3)).expand((-1,-1,3))
            scale = torch.round(block_delta[:,:,1]/box_vectors[1,1])
            block_delta = block_delta - scale.unsqueeze(2)*box_vectors[1].view((1,1,3)).expand((-1,-1,3))
            scale = torch.round(block_delta[:,:,0]/box_vectors[0,0])
            block_delta = block_delta - scale.unsqueeze(2)*box_vectors[0].view((1,1,3)).expand((-1,-1,3))
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
