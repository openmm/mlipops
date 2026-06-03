import torch
from collections.abc import Callable
from typing import Any
from .utils import periodic_displacements
try:
    import triton
    from .pairwise_triton import backprop_delta_kernel
    has_triton = True
except ImportError:
    has_triton = False


class Pairwise(torch.nn.Module):
    """Computes pairwise interactions between particles.

    This class can be used to implement arbitrary interactions of the form

    .. math::
        E = \\sum_{i,j} f(r_{ij}, P_i, P_j)

    where f is a function you provide, :math:`r_{ij}` is the distance between particles i and j, and :math:`P_i`
    is a vector of per-particle parameters for particle i.  To use it, first define a function to compute the
    interaction.  For example,

    >>> def coulomb(pairs, r, params):
    >>>     return params[pairs[:,0]]*params[pairs[:,1]]/r

    The function should take three arguments.  `pairs` is a Tensor of shape (n_pairs, 2), with each row containing
    the indices of two interacting particles.  Typically it is computed by a NeighborList.  `r` is a Tensor of
    shape (n_pairs,) containing the distance between each pair of interacting particles.  `params` is an arbitrary
    object containing additional parameters on which the interaction can depend.  In the above example, it should
    be a Tensor of shape (n_particles,) containing the charge on each particle.

    Next create a Pairwise object, passing your function to the constructor.

    >>> pairwise = Pairwise(coulomb)

    And finally evaluate it, passing in the particle positions, parameters, pairs, and periodic box vectors.

    >>> energy = pairwise(positions, params, pairs, box_vectors)

    When creating a Pairwise object, you can optionally specify a cutoff distance.  Pairs of particles that are
    further apart than the cutoff are ignored.  This is useful when using a NeighborList with padding, so some
    of the returned pairs are beyond the cutoff.  You also can specify a list of specific particle pairs whose
    interaction should always be excluded, regardless of their distance.
    """
    def __init__(self, computation: Callable, cutoff: float | None, exclusions: torch.Tensor | None = None):
        """Create an object for computing pairwise interactions.

        Parameters
        __________
        computation: Callable
            a callable object that defines how the interaction is computed
        cutoff: float | None
            if specified, any pair whose distance is greater than the cutoff will be omitted
        exclusions: torch.Tensor
            a tensor of shape (n_exclusions, 2).  Each row contains the indices of two particles whose interaction
            should always be omitted.
        """
        super().__init__()
        self.computation = computation
        self.cutoff = cutoff
        self.register_buffer('exclusions', exclusions)
        if exclusions is None:
            self.exclusion_indices = None
        else:
            indices = [[] for _ in range(exclusions.max()+1)]
            for pair in exclusions:
                p1, p2 = int(pair[0]), int(pair[1])
                indices[p1].append(p2)
                indices[p2].append(p1)
            max_exclusions = max(len(i) for i in indices)
            exclusion_indices = -1*torch.ones((len(indices), max_exclusions+1), dtype=torch.int32, device=exclusions.device)
            for i in range(len(indices)):
                for j in range(len(indices[i])):
                    exclusion_indices[i][j] = indices[i][j]
            self.register_buffer('exclusion_indices', exclusion_indices)

    def forward(self, positions: torch.Tensor, parameters: Any, pairs: torch.Tensor, box_vectors: torch.Tensor | None):
        """Compute the interaction.

        Parameters
        ----------
        positions: torch.Tensor
            a Tensor of shape (n_particles, 3) containing the cartesian coordinates of each particle
        parameters: Any
            an arbitrary object containing parameter values.  It is passed to the computation function.
        pairs: torch.Tensor
            a Tensor of shape (n_pairs, 2).  Each row contains the indices of two particles that interact.  Typically
            it is created by a NeighborList.
        box_vectors: torch.Tensor | None
            a Tensor of shape (3, 3) containing box vectors defining the periodic box.  If None, periodic boundary
            conditions are not used.

        Returns
        -------
        a torch.Tensor containing the energy of the interaction
        """
        if has_triton and positions.device.type == 'cuda':
            delta = DeltaFunction.apply(positions, pairs, box_vectors)
        else:
            delta = periodic_displacements(positions[pairs[:,1]] - positions[pairs[:,0]], box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        energy = self.computation(pairs, distance, parameters)
        masks = []
        if self.cutoff is not None:
            masks.append(distance < self.cutoff)
        if self.exclusion_indices is not None:
            if positions.shape[0] > self.exclusion_indices.shape[0]:
                padding = positions.shape[0] - self.exclusion_indices.shape[0]
                self.exclusion_indices = torch.nn.functional.pad(self.exclusion_indices, pad=(0,0,0,padding), value=-1)
            masks.append(~torch.any(self.exclusion_indices[pairs[:,0]] == pairs[:,1].reshape((-1,1)), dim=1))
        if len(masks) == 0:
            return torch.sum(energy)
        if len(masks) == 1:
            mask = masks[0]
        else:
            mask = masks[0]*masks[1]
        return torch.sum(torch.where(mask, energy, 0.0))


class DeltaFunction(torch.autograd.Function):
    """Compute the displacement between pairs of particles, optionally taking periodic boundary conditions into
    account.  PyTorch can compute the forward pass efficiently, but the default implementation of the backward pass
    is very slow.  We use a Triton kernel to do it more efficiently.
    """
    @staticmethod
    def forward(ctx, positions: torch.Tensor, pairs: torch.Tensor, box_vectors: torch.Tensor):
        delta = periodic_displacements(positions[pairs[:,1]] - positions[pairs[:,0]], box_vectors)
        ctx.save_for_backward(positions, pairs)
        return delta

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        positions, pairs = ctx.saved_tensors
        result = torch.zeros_like(positions)
        g = lambda meta: (triton.cdiv(positions.shape[0], meta['BLOCK_SIZE']),)
        backprop_delta_kernel[g](result, grad_outputs[0], pairs, pairs.shape[0], 256)
        return result, None, None
