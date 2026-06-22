import math
import torch
from collections.abc import Callable


def point_charge_interaction(pairs, r, delta, charge):
    """Compute the Coulomb interaction between point charges.  This function is designed for use with Pairwise."""
    return charge[pairs[:,0]]*charge[pairs[:,1]]/r


def dipole_interaction(pairs, r, delta, params):
    """Compute the Coulomb interaction between multipoles, each having a charge and dipole moment.  This function is
    designed for use with Pairwise."""
    charge, dipole = params
    p1 = pairs[:,0]
    p2 = pairs[:,1]
    c1 = charge[p1]
    c2 = charge[p2]
    d1 = dipole[p1]
    d2 = dipole[p2]
    d1delta = (d1*delta).sum(axis=1)
    d2delta = (d2*delta).sum(axis=1)
    denom3 = r**-3
    energy_cc = c1*c2/r
    energy_cd = (c2*d1delta - c1*d2delta)*denom3
    energy_dd = ((d1*d2).sum(axis=1) - 3*d1delta*d2delta*r**-2)*denom3
    return energy_cc + energy_cd + energy_dd


class ErfcScaledInteraction(object):
    """Scale an interaction function by erfc(alpha*r), where alpha is a constant.  This corresponds to the direct space
    part of Ewald summation.  This is a callable object designed for use with Pairwise."""
    def __init__(self, computation: Callable, alpha: float):
        self.computation = computation
        self.alpha = alpha

    def __call__(self, pairs, r, delta, params):
        return torch.erfc(self.alpha*r)*self.computation(pairs, r, delta, params)


class ErfScaledInteraction(object):
    """Scale an interaction function by erf(alpha*r), where alpha is a constant.  This corresponds to the reciprocal
    space part of Ewald summation.  This is a callable object designed for use with Pairwise."""
    def __init__(self, computation: Callable, alpha: float):
        self.computation = computation
        self.alpha = alpha

    def __call__(self, pairs, r, delta, params):
        return torch.erf(self.alpha*r)*self.computation(pairs, r, delta, params)


class ReactionFieldInteraction(object):
    """Compute the Coulomb interaction between point charges using reaction field.  This function is designed for use
    with Pairwise."""
    def __init__(self, cutoff: float, dielectric: float):
        self.k = (cutoff**-3)*(dielectric-1)/(2*dielectric+1)
        self.c = 3*dielectric/(cutoff*(2*dielectric+1))

    def __call__(self, pairs, r, delta, params):
        return params[pairs[:,0]]*params[pairs[:,1]]*(1/r + self.k*r**2 - self.c)


class ErfcScaledDipoleInteraction(object):
    """Compute the Coulomb interaction between multipoles, each having a charge and dipole moment, with the strength
    scaled by erfc(alpha*r).  This corresponds to the direct space part of Ewald summation.  This is a callable object
    designed for use with Pairwise."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.temp1 = 2*alpha/math.sqrt(math.pi)
        self.temp2 = 4*alpha**3/math.sqrt(math.pi)

    def __call__(self, pairs, r, delta, params):
        charge, dipole = params
        p1 = pairs[:,0]
        p2 = pairs[:,1]
        c1 = charge[p1]
        c2 = charge[p2]
        d1 = dipole[p1]
        d2 = dipole[p2]
        d1delta = (d1*delta).sum(axis=1)
        d2delta = (d2*delta).sum(axis=1)
        alphar = self.alpha*r
        rinv2 = r**-2
        expfactor = torch.exp(-alphar**2)

        # See equations 2.8 and 2.10 of https://doi.org/10.1063/1.1324708.

        b0 = torch.erfc(alphar)/r
        b1 = rinv2*(b0 + self.temp1*expfactor)
        b2 = rinv2*(3*b1 + self.temp2*expfactor)
        g0 = c1*c2
        g1 = c2*d1delta - c1*d2delta + (d1*d2).sum(axis=1)
        g2 = -d1delta*d2delta
        return (b0*g0 + b1*g1 + b2*g2)