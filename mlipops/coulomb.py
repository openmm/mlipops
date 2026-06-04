import torch
from collections.abc import Callable

def point_charge_interaction(pairs, r, params):
    """Compute the Coulomb interaction between point charges.  This function is designed for use with Pairwise."""
    return params[pairs[:,0]]*params[pairs[:,1]]/r

class ErfcScaledInteraction(object):
    """Scale an interaction function by erfc(alpha*r), where alpha is a constant.  This corresponds to the direct space
    part of Ewald summation.  This is a callable object designed for use with Pairwise."""
    def __init__(self, computation: Callable, alpha: float):
        self.computation = computation
        self.alpha = alpha

    def __call__(self, pairs, r, params):
        return torch.erfc(self.alpha*r)*self.computation(pairs, r, params)

class ErfScaledInteraction(object):
    """Scale an interaction function by erf(alpha*r), where alpha is a constant.  This corresponds to the reciprocal
    space part of Ewald summation.  This is a callable object designed for use with Pairwise."""
    def __init__(self, computation: Callable, alpha: float):
        self.computation = computation
        self.alpha = alpha

    def __call__(self, pairs, r, params):
        return torch.erf(self.alpha*r)*self.computation(pairs, r, params)


class ReactionFieldInteraction(object):
    """Compute the Coulomb interaction between point charges using reaction field.  This function is designed for use
    with Pairwise."""
    def __init__(self, cutoff: float, dielectric: float):
        self.k = (cutoff**-3)*(dielectric-1)/(2*dielectric+1)
        self.c = 3*dielectric/(cutoff*(2*dielectric+1))

    def __call__(self, pairs, r, params):
        return params[pairs[:,0]]*params[pairs[:,1]]*(1/r + self.k*r**2 - self.c)
