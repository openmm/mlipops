import torch

def coulomb_no_cutoff(pairs, distance, parameters):
    return parameters[:,0,0]*parameters[:,1,0]/distance

class CoulombEwald(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, pairs, distance, parameters):
        return torch.erfc(self.alpha*distance)*coulomb_no_cutoff(pairs, distance, parameters)

class CoulombEwaldExclusionCorrection(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, pairs, distance, parameters):
        return torch.erf(self.alpha*distance)*coulomb_no_cutoff(pairs, distance, parameters)
