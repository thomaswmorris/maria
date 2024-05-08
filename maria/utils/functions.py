import numpy as np
import scipy as sp

from maria.constants import c, h, k_B


def planck_spectrum(nu, T):
    return 2 * h * nu**3 / (c**2 * np.expm1(h * nu / (k_B * T)))


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


def approximate_normalized_matern(r_eff, nu, n_test_points=1024):
    """
    Computing BesselK[nu,z] for arbitrary nu is expensive. This is good for casting over huge matrices.
    """

    r_min, r_max = 1e-6, 1e3
    r_eff_safe = np.atleast_1d(np.abs(r_eff)).clip(min=r_min, max=r_max)
    r_samples = np.geomspace(r_min, r_max, n_test_points)
    cov_samples = normalized_matern(r_samples, nu=nu)

    with np.errstate(divide="ignore"):
        sf = np.exp(
            np.interp(np.log(r_eff_safe), np.log(r_samples), np.log(1 - cov_samples))
        )
        cov = np.exp(
            np.interp(np.log(r_eff_safe), np.log(r_samples), np.log(cov_samples))
        )

    # we combine the log interpolations so that both extremes have really good precision
    # the structure function and covariance are equal at around 1, so that's our inflection point
    t = 1 / (1 + r_eff_safe**2)

    return t * (1 - sf) + (1 - t) * cov
