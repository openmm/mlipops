import torch

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
