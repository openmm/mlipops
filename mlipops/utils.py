import torch
try:
    import triton
    from .utils_triton import backprop_delta_kernel
    has_triton = True
except ImportError:
    has_triton = False


def periodic_displacements(displacements: torch.Tensor, box_vectors: torch.Tensor | None):
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


def pairwise_displacements(positions: torch.Tensor, pairs: torch.Tensor, box_vectors: torch.Tensor | None):
    """Compute the displacement between pairs of points, optionally taking periodic boundary conditions into
    account.

    Parameters
    ----------
    positions: torch.Tensor
        a Tensor of shape (n_point, 3) containing the Cartesian coordinates of each point
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
    if has_triton and positions.device.type == 'cuda':
        return DisplacementFunction.apply(positions, pairs, box_vectors)
    return periodic_displacements(positions[pairs[:,1]] - positions[pairs[:,0]], box_vectors)


class DisplacementFunction(torch.autograd.Function):
    """Compute the displacement between pairs of points, optionally taking periodic boundary conditions into
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
