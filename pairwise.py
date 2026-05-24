import torch
from collections.abc import Callable

class Pairwise(torch.nn.Module):
    def __init__(self, torch_computation: Callable, cutoff: float, exclusions: torch.Tensor | None = None):
        super().__init__()
        self.torch_computation = torch_computation
        self.cutoff = cutoff
        self.register_buffer('exclusions', exclusions)
        if exclusions is None:
            exclusion_indices = None
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

    def forward(self, positions: torch.Tensor, parameters: torch.Tensor | None, pairs: torch.Tensor, box_vectors: torch.Tensor | None):
        delta = positions[pairs[:,1]] - positions[pairs[:,0]]
        if box_vectors is not None:
            scale = torch.round(delta[:,2]/box_vectors[2,2])
            delta = delta - scale.unsqueeze(1)*box_vectors[2].view((1,3)).expand((-1,3))
            scale = torch.round(delta[:,1]/box_vectors[1,1])
            delta = delta - scale.unsqueeze(1)*box_vectors[1].view((1,3)).expand((-1,3))
            scale = torch.round(delta[:,0]/box_vectors[0,0])
            delta = delta - scale.unsqueeze(1)*box_vectors[0].view((1,3)).expand((-1,3))
        distance = torch.linalg.vector_norm(delta, dim=1)
        parameters = parameters[pairs] if parameters is not None else None
        energy = self.torch_computation(pairs, distance, parameters)
        mask = distance < self.cutoff
        if self.exclusion_indices is not None:
            mask *= ~torch.any(self.exclusion_indices[pairs[:,0]] == pairs[:,1].reshape((-1,1)), dim=1)
        return torch.sum(torch.where(mask, energy, 0.0))