import torch

def coulomb_no_cutoff(pairs, r, params):
    return params[pairs[:,0]]*params[pairs[:,1]]/r

class CoulombEwald(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, pairs, r, params):
        return torch.erfc(self.alpha*r)*coulomb_no_cutoff(pairs, r, params)

class CoulombEwaldExclusionCorrection(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, pairs, r, params):
        return torch.erf(self.alpha*r)*coulomb_no_cutoff(pairs, r, params)
