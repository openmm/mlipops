import torch
from collections.abc import Callable
try:
    import triton
    from pairwise_triton import backprop_delta_kernel
    has_triton = True
except ImportError:
    has_triton = False


class Pairwise(torch.nn.Module):
    def __init__(self, torch_computation: Callable, cutoff: float | None, exclusions: torch.Tensor | None = None):
        super().__init__()
        self.torch_computation = torch_computation
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

    def forward(self, positions: torch.Tensor, parameters: torch.Tensor | None, pairs: torch.Tensor, box_vectors: torch.Tensor | None):
        if has_triton and positions.device.type == 'cuda':
            delta = DeltaFunction.apply(positions, pairs, box_vectors)
        else:
            delta = periodic_delta(positions, pairs, box_vectors)
        distance = torch.linalg.vector_norm(delta, dim=1)
        parameters = parameters[pairs] if parameters is not None else None
        energy = self.torch_computation(pairs, distance, parameters)
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


def periodic_delta(positions: torch.Tensor, pairs: torch.Tensor, box_vectors: torch.Tensor):
    delta = positions[pairs[:,1]] - positions[pairs[:,0]]
    if box_vectors is not None:
        scale = torch.round(delta[:,2]/box_vectors[2,2])
        delta = delta - scale.unsqueeze(1)*box_vectors[2].view((1,3)).expand((-1,3))
        scale = torch.round(delta[:,1]/box_vectors[1,1])
        delta = delta - scale.unsqueeze(1)*box_vectors[1].view((1,3)).expand((-1,3))
        scale = torch.round(delta[:,0]/box_vectors[0,0])
        delta = delta - scale.unsqueeze(1)*box_vectors[0].view((1,3)).expand((-1,3))
    return delta


class DeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions: torch.Tensor, pairs: torch.Tensor, box_vectors: torch.Tensor):
        delta = periodic_delta(positions, pairs, box_vectors)
        ctx.save_for_backward(positions, pairs)
        return delta

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        positions, pairs = ctx.saved_tensors
        result = torch.zeros_like(positions)
        g = lambda meta: (triton.cdiv(positions.shape[0], meta['BLOCK_SIZE']),)
        backprop_delta_kernel[g](result, grad_outputs[0], pairs, pairs.shape[0], 256)
        return result, None, None
