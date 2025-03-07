import numpy as np

from ..constants import c, h, k_B  # noqa


def rayleigh_jeans_spectrum(T_RJ: float, nu: float):
    return 2 * k_B * nu**2 * T_RJ / c**2


def inverse_rayleigh_jeans_spectrum(I_nu: float, nu: float):
    return I_nu * c**2 / (2 * k_B * nu**2)


def planck_spectrum(T_b: float, nu: float):
    return 2 * h * nu**3 / (c**2 * np.expm1(h * nu / (k_B * T_b)))


def inverse_planck_spectrum(I_nu: float, nu: float):
    return (h * nu / k_B) / np.log1p(2 * h * nu**3 / (I_nu * c**2))
