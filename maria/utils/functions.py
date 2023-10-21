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


def approximate_normalized_matern(r, nu, n_test_points=1024):
    """
    Computing BesselK[nu,z] for arbitrary nu is expensive. This is good for casting over huge matrices.
    """

    r_safe = np.atleast_1d(np.abs(r))
    r_min = np.maximum(r_safe[r_safe > 0].min(), 1e-6)
    r_max = np.minimum(r_safe.max(), 1e3)

    test_values = np.r_[0, np.geomspace(r_min, r_max, n_test_points - 1)]
    data_values = normalized_matern(test_values, nu) * np.exp(test_values)

    return np.interp(r_safe, test_values, data_values) * np.exp(-r_safe)
