import os

import numpy as np
import pandas as pd

from .quantities import parse_units, QUANTITIES
from ..constants import k_B, T_CMB
from ..functions.radiometry import (
    rayleigh_jeans_spectrum,
    inverse_rayleigh_jeans_spectrum,
    planck_spectrum,
    inverse_planck_spectrum,
)  # noqa

here, this_filename = os.path.split(__file__)


def identity(x: float, **kwargs):
    return x


def rayleigh_jeans_temperature_to_brightness_temperature(T_RJ, nu, **kwargs):
    I_nu = rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu)
    return inverse_planck_spectrum(I_nu=I_nu, nu=nu)


def brightness_temperature_to_rayleigh_jeans_temperature(T_b, nu, **kwargs):
    I_nu = planck_spectrum(T_b=T_b, nu=nu)
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def rayleigh_jeans_temperature_to_radiant_flux(T_RJ, passband, **kwargs):
    """
    nu: frequency, in Hz
    passband: response to a Rayleigh-Jeans source
    """
    return T_RJ * k_B * np.trapezoid(y=passband["tau"], x=passband["nu"])


def radiant_flux_to_rayleigh_jeans_temperature(P, passband, **kwargs):
    """
    nu: frequency, in Hz
    passband: response to a Rayleigh-Jeans source
    """
    return P / (k_B * np.trapezoid(y=passband["tau"], x=passband["nu"]))


def rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel(
    T_RJ: float, nu: float, res: float, **kwargs
):
    """
    T_RJ: Rayleigh-Jeans temperature, in Kelvin
    nu: frequency, in Hz
    res: resolution, in radians
    """
    return 1e26 * rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu) * res**2


def spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature(
    E: float, nu: float, res: float, **kwargs
):
    """
    T_RJ: Rayleigh-Jeans temperature, in Jy/pixel
    nu: frequency, in Hz
    res: resolution, in radians
    """
    I_nu = 1e-26 * E / res**2
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def cmb_temperature_anisotropy_to_radiant_flux_slope(passband: dict, eps: float = 1e-3):
    test_T_b = T_CMB + np.array([[-eps / 2], [+eps / 2]])
    T_RJ = inverse_rayleigh_jeans_spectrum(
        planck_spectrum(T_b=test_T_b, nu=passband["nu"]), nu=passband["nu"]
    )
    P = k_B * np.trapezoid(T_RJ * passband["tau"], x=passband["nu"])
    return (P[1] - P[0]) / eps


def cmb_temperature_anisotropy_to_rayleigh_jeans_temperature(
    delta_T: float, passband: dict
):
    dP_dTCMB = cmb_temperature_anisotropy_to_radiant_flux_slope(passband=passband)
    return radiant_flux_to_rayleigh_jeans_temperature(
        dP_dTCMB * delta_T, passband=passband
    )


def rayleigh_jeans_temperature_to_cmb_temperature_anisotropy(
    T_RJ: float, passband: dict
):
    dP_dTCMB = cmb_temperature_anisotropy_to_radiant_flux_slope(passband=passband)
    return (
        rayleigh_jeans_temperature_to_radiant_flux(T_RJ, passband=passband) / dP_dTCMB
    )


QUANTITIES.loc["rayleigh_jeans_temperature", "to"] = identity
QUANTITIES.loc["rayleigh_jeans_temperature", "from"] = identity

QUANTITIES.loc["brightness_temperature", "to"] = (
    rayleigh_jeans_temperature_to_brightness_temperature
)
QUANTITIES.loc["brightness_temperature", "from"] = (
    brightness_temperature_to_rayleigh_jeans_temperature
)

QUANTITIES.loc["cmb_temperature_anisotropy", "to"] = (
    rayleigh_jeans_temperature_to_cmb_temperature_anisotropy
)
QUANTITIES.loc["cmb_temperature_anisotropy", "from"] = (
    cmb_temperature_anisotropy_to_rayleigh_jeans_temperature
)

QUANTITIES.loc["radiant_flux", "to"] = rayleigh_jeans_temperature_to_radiant_flux
QUANTITIES.loc["radiant_flux", "from"] = radiant_flux_to_rayleigh_jeans_temperature

QUANTITIES.loc["spectral_radiance", "to"] = inverse_rayleigh_jeans_spectrum
QUANTITIES.loc["spectral_radiance", "from"] = rayleigh_jeans_spectrum

QUANTITIES.loc["spectral_flux_density_per_pixel", "to"] = (
    rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel
)
QUANTITIES.loc["spectral_flux_density_per_pixel", "from"] = (
    spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature
)


def parse_calibration_signature(s: str):
    res = {}
    for sep in ["->"]:
        if s.count(sep) == 1:
            if sep is not None:
                items = [u.strip() for u in s.split(sep)]
                if len(items) == 2:
                    for io, u in zip(["in", "out"], items):
                        res[io] = parse_units(u)
        return res
    raise ValueError("Calibration must have signature 'units1 -> units2'.")


class Calibration:

    def __init__(self, signature: str, **kwargs):

        self.config = pd.DataFrame(parse_calibration_signature(signature))
        self.signature = signature
        self.kwargs = kwargs

        for key, value in kwargs.items():
            if key not in ["nu", "res", "passband"]:
                raise ValueError(f"Invalid kwarg '{key}'.")

    def __call__(self, x) -> float:
        return (
            self.K_RJ_to_out(
                self.in_to_K_RJ(x * self.in_factor, **self.kwargs), **self.kwargs
            )
            / self.out_factor
        )

    @property
    def in_factor(self) -> float:
        return self.config.loc["factor", "in"]

    @property
    def out_factor(self) -> float:
        return self.config.loc["factor", "out"]

    @property
    def in_to_K_RJ(self) -> float:
        return self.config.loc["from", "in"]

    @property
    def K_RJ_to_out(self) -> float:
        return self.config.loc["to", "out"]

    def __repr__(self):
        stuffing = ", ".join(
            [self.signature, *[f"{k}={v}" for k, v in self.kwargs.items()]]
        )
        return f"Calibration({stuffing})"
