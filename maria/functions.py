import numpy as np
import scipy as sp

from maria.units.constants import c, h, k_B


def rayleigh_jeans_spectrum(nu: float, T: float):
    return 2 * k_B * nu**2 * T / c**2


def planck_spectrum(nu: float, T: float):
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
        * sp.special.kv(nu, np.sqrt(2 * nu) * r + 1e-16)
        * (np.sqrt(2 * nu) * r + 1e-16) ** nu
    )


def approximate_normalized_matern(r, nu=1 / 3, r0=1e0, n_test_points=1024):
    """
    Computing BesselK[nu,z] for arbitrary nu is expensive. This is good for casting over huge matrices.
    """
    r_eff = r / r0
    r_eff_min, r_eff_max = 1e-6, 1e3

    r_eff_safe = np.atleast_1d(np.abs(r_eff)).clip(min=r_eff_min)
    r_eff_nonzero = r_eff_safe[r_eff_safe < r_eff_max]

    r_eff_samples = np.geomspace(r_eff_min, r_eff_max, n_test_points)
    cov_samples = normalized_matern(r_eff_samples, nu=nu)

    with np.errstate(divide="ignore"):
        sf = np.exp(
            np.interp(
                np.log(r_eff_nonzero), np.log(r_eff_samples), np.log(1 - cov_samples)
            )
        )
        cov = np.exp(
            np.interp(np.log(r_eff_nonzero), np.log(r_eff_samples), np.log(cov_samples))
        )

    # we combine the log interpolations so that both extremes have really good precision
    # the structure function and covariance are equal at around 1, so that's our inflection point
    t = 1 / (1 + r_eff_nonzero**2)

    res = np.zeros(r.shape)
    res[r_eff_safe < r_eff_max] = t * (1 - sf) + (1 - t) * cov

    return res
