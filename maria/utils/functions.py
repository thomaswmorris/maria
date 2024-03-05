import numpy as np
import scipy as sp


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(y):
    return -np.log(1 / y - 1)


def matern(r, r0, nu):
    """
    Matern covariance
    """
    return normalized_matern(r / r0, nu)


def normalized_matern(r, nu):
    """
    Matern covariance
    """
    return (
        2 ** (1 - nu)
        / sp.special.gamma(nu)
        * sp.special.kv(nu, r + 1e-16)
        * (r + 1e-16) ** nu
    )


def approximate_normalized_matern(r, nu, n_test_points=256):
    """
    Computing BesselK[nu,z] for arbitrary nu is expensive. This is good for casting over huge matrices.
    """

    r_safe = np.atleast_1d(np.abs(r))
    r_min = np.maximum(1e-6, 0.5 * r_safe[r_safe > 0].min())
    r_max = np.minimum(1e3, 2.0 * r_safe.max())

    r_samples = np.geomspace(r_min, r_max, n_test_points)

    sf_samples = 1 - normalized_matern(r_samples, nu=nu)

    sf = np.exp(np.interp(np.log(r), np.log(r_samples), np.log(sf_samples)))

    return 1 - sf
