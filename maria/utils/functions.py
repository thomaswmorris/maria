import numpy as np
import scipy as sp

def gaussian_beam_waist(z, w0, l, n): 
    return np.sqrt(1 / np.square(z) + np.square(l) / np.square(w0 * np.pi * n))

def matern(r, r0, nu):
    """
    Matern covariance
    """
    return 2 ** (1 - nu) / sp.special.gamma(nu) * sp.special.kv(nu, r / r0 + 1e-16) * (r / r0 + 1e-16) ** nu

def sigmoid(x): 
    return 1/(1+np.exp(-x))

def inverse_sigmoid(y):
    return -np.log(1/y - 1)

def approximate_matern(r, r0, nu, n_test_points=4096):
    """
    Computing BesselK[nu,z] for arbitrary nu is expensive. This is good for casting over huge matrices.
    """
    
    r_eff = np.atleast_1d(r / r0)

    if not r_eff.min() >= 0:
        raise ValueError()

    r_min = r_eff[r_eff > 0].min()
    r_max = r_eff.max()                       
    r_test = np.r_[0, np.geomspace(r_min, r_max, n_test_points-1)]
    
    return np.exp(np.interp(r, r_test, np.log(matern(r_test, 1, nu))))
