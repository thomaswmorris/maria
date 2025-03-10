from __future__ import annotations

import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from ..atmosphere import AtmosphericSpectrum
from ..calibration import Calibration
from ..constants import c
from ..io import humanize
from ..utils import flatten_config, read_yaml

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

BAND_FIELD_FORMATS = pd.read_csv(f"{here}/format.csv", index_col=0)

BAND_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    BAND_CONFIGS[tag] = read_yaml(path)

BAND_CONFIGS = flatten_config(BAND_CONFIGS)

band_data = pd.DataFrame(BAND_CONFIGS).T.sort_index()
all_bands = list(band_data.index)


def get_band(band_name):
    if band_name in BAND_CONFIGS:
        return Band(name=band_name, **BAND_CONFIGS[band_name])
    else:
        raise ValueError(f"'{band_name}' is not a valid pre-defined band name.")


def generate_passband(center, width, shape, samples=256):
    if shape == "flat":
        nu_min, nu_max = (center - 0.6 * width, center + 0.6 * width)
    else:
        nu_min, nu_max = (center - 1.5 * width, center + 1.5 * width)

    nu = np.linspace(nu_min, nu_max, samples)

    if shape == "flat":
        tau = np.where((nu > center - 0.5 * width) & (nu < center + 0.5 * width), 1, 0)
    elif shape == "gaussian":
        tau = np.exp(np.log(0.5) * (2 * (nu - center) / width) ** 2)
    elif shape == "top_hat":
        tau = np.exp(np.log(0.5) * (2 * (nu - center) / width) ** 8)
    else:
        raise ValueError(f"Invalid shape '{shape}'.")
    return nu, tau


class Band:
    def __init__(
        self,
        center: float = None,
        width: float = None,
        nu: float = None,
        tau: float = None,
        name: str = None,
        shape: str = "top_hat",
        efficiency: float = 0.5,
        sensitivity: float = None,
        NET_RJ: float = None,
        NET_CMB: float = None,
        NEP: float = None,
        NEP_per_loading: float = 0.0,
        gain_error: float = 0,
        knee: float = 1.0,
        time_constant: float = 0.0,
        spectrum_kwargs: dict = {},
    ):
        auto = center is not None and width is not None
        manual = nu is not None and tau is not None

        if not auto ^ manual:
            raise ValueError(
                "You must pass either both 'center' and 'width' or both 'nu' and 'tau'.",
            )

        if auto:
            self.nu, self.tau = generate_passband(center, width, shape, samples=1024)

        if manual:
            tau_max = np.max(tau)
            efficiency *= tau_max

            self.nu = np.array(nu)
            self.tau = np.array(tau) / tau_max

            if (self.nu.ndim != 1) or (self.tau.ndim != 1) or (self.nu.shape != self.tau.shape):
                raise ValueError(
                    f"'nu' and 'tau' have mismatched shapes ({self.nu.shape} and {self.tau.shape}).",
                )

        self._name = name
        self.shape = shape
        self.efficiency = efficiency
        self.NEP_per_loading = NEP_per_loading
        self.knee = knee
        self.time_constant = time_constant
        self.gain_error = gain_error
        self.spectrum_kwargs = spectrum_kwargs
        self.spectrum = AtmosphericSpectrum(region=spectrum_kwargs) if spectrum_kwargs else None

        if sensitivity:
            logger.warning(
                "The 'sensitivity' keyword is deprecated and will be removed in future releases. To specify noise levels in terms of sky temperature, use the 'NET_RJ' or 'NET_CMB' keywords instead."
            )
            NET_RJ = sensitivity

        if (NEP is None) and (NET_RJ is None) and (NET_CMB is None):
            logger.warning(f"No noise level specified for band {self.name}, assuming a sensitivity of 1 uK.")
            self.NET_RJ = 1e-6

        else:
            if NEP is not None:
                self.NEP = NEP
            elif NET_RJ is not None:
                self.NET_RJ = NET_RJ
            elif NET_CMB is not None:
                self.NET_CMB = NET_CMB

        self.transmission_integral_grids = {}

    @property
    def default_spectrum_kwargs(self):
        if self.spectrum is not None:
            return {"zenith_pwv": 1e0, "elevation": 90, "base_temperature": self.spectrum.side_base_temperature.mean()}
        return {}

    @property
    def name(self):
        return self._name or f"f{int(self.center):>03}"

    @property
    def center(self):
        return np.round(np.sum(self.nu * self.tau) / np.sum(self.tau), 2)

    @property
    def width(self):
        return np.round(self.fwhm, 2)

    @property
    def fwhm(self):
        crossovers = np.where((self.tau[1:] > 0.5) != (self.tau[:-1] > 0.5))[0]
        return np.ptp(
            [sp.interpolate.interp1d(self.tau[[i, i + 1]], self.nu[[i, i + 1]])(0.5) for i in crossovers],
        )

    def summary(self):
        summary = pd.Series(index=BAND_FIELD_FORMATS.index, dtype=str)

        for field, entry in BAND_FIELD_FORMATS.iterrows():
            value = getattr(self, field)

            if (entry["units"] != "none") and (entry["dtype"] == "float"):
                s = humanize(value, unit=entry["units"])
            elif entry["dtype"] == "str":
                s = f"'{value}'"
            else:
                s = f"{value}"

            summary[field] = s

        return summary

    def __repr__(self):
        return f"Band({', '.join([f'{index}={entry}' for index, entry in self.summary().items()])})"

    def plot(self):
        fig, ax = plt.subplots(1, 1)

        ax.plot(self.nu, self.tau, label=self.name)

        ax.set_xlabel(r"$\nu$ [GHz]")
        ax.set_ylabel(r"$\tau(\nu)$ [Rayleigh-Jeans]")
        ax.legend()

    @property
    def NET_RJ(self):
        return self.cal("W -> K_RJ", spectrum=self.spectrum, **self.default_spectrum_kwargs)(self.NEP).item()

    @NET_RJ.setter
    def NET_RJ(self, value):
        self.NEP = self.cal("K_RJ -> W", spectrum=self.spectrum, **self.default_spectrum_kwargs)(value).item()

    @property
    def NET_CMB(self):
        return self.cal("W -> K_CMB", spectrum=self.spectrum, **self.default_spectrum_kwargs)(self.NEP).item()

    @NET_CMB.setter
    def NET_CMB(self, value):
        self.NEP = self.cal("K_CMB -> W", spectrum=self.spectrum, **self.default_spectrum_kwargs)(value).item()

    def compute_nu_integral(
        self,
        spectrum: AtmosphericSpectrum = None,
        nu_min: float = 0,
        nu_max: float = np.inf,
        **kwargs,
    ):
        """
        This is only useful for Rayleigh-Jeans sources (i.e. with linear emission).
        """

        if spectrum is None:
            nu = self.nu[(self.nu >= nu_min) & (self.nu < nu_max)]
            return np.trapezoid(y=self.passband(nu), x=1e9 * nu, axis=-1)

        else:
            nu = spectrum.side_nu[(spectrum.side_nu >= nu_min) & (spectrum.side_nu < nu_max)]
            integral_grid = np.trapezoid(y=self.passband(nu) * np.exp(-spectrum._opacity), x=1e9 * nu, axis=-1)
            xi = (kwargs["zenith_pwv"], kwargs["base_temperature"], kwargs["elevation"])
            return sp.interpolate.interpn(points=spectrum.points[:3], values=integral_grid, xi=xi)

    def transmission(self, region="chajnantor", pwv=1, elevation=90) -> float:
        if not hasattr(self, "spectrum"):
            self.spectrum = AtmosphericSpectrum(region=region)
        elif self.spectrum.region != region:
            self.spectrum = AtmosphericSpectrum(region=region)
        return self.spectrum.transmission(nu=self.center, pwv=pwv, elevation=elevation)

    # @classmethod
    # def from_file(cls, filename, **kwargs):
    #     path = (
    #         f"{here}/{filename}" if os.path.exists(f"{here}/{filename}") else filename
    #     )

    #     df = pd.read_csv(path, index_col=0)

    #     nu, tau = df.nu.values, df.passband.values

    #     # dummy values for center and width
    #     band = cls(nu=nu, tau=tau, shape="custom", **kwargs)
    #     band.nu = nu
    #     band.tau = tau

    def passband(self, nu):
        return self.efficiency * sp.interpolate.interp1d(
            self.nu,
            self.tau,
            bounds_error=False,
            fill_value=0,
        )(nu)

    # @property
    # def dP_dTRJ(self) -> float:
    #     """
    #     In watts per Kelvin Rayleigh-Jeans
    #     """
    #     T_0 = 1e0
    #     eps = 1e-3

    #     return (self.cal("K_RJ -> W")(T_0 + eps) - self.cal("K_RJ -> W")(T_0)) / eps

    # @property
    # def dP_dTCMB(self) -> float:
    #     """
    #     In watts per kelvin CMB, assuming perfect transmission.
    #     """

    #     eps = 1e-3
    #     delta_T = np.array([-eps / 2, eps / 2])

    #     TRJ = (
    #         planck_spectrum(nu=1e9 * self.nu, T=T_CMB + delta_T[:, None])
    #         * c**2
    #         / (2 * k_B * (1e9 * self.nu) ** 2)
    #     )

    #     return (
    #         self.efficiency
    #         * k_B
    #         * np.diff(np.trapezoid(TRJ * self.passband(self.nu), 1e9 * self.nu))[0]
    #         / eps
    #     )

    @property
    def wavelength(self):
        """
        Return the wavelength of the center, in meters.
        """
        return c / (1e9 * self.center)

    def cal(self, signature: str, **kwargs) -> float:
        return Calibration(
            signature,
            band=self,
            **kwargs,
        )
