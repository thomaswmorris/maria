from __future__ import annotations

import glob
import logging
import os
from collections.abc import Mapping

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from jax import scipy as jsp

from ..atmosphere import AtmosphericSpectrum
from ..calibration import Calibration
from ..constants import MARIA_MAX_NU, MARIA_MIN_NU, c
from ..errors import FrequencyOutOfBoundsError
from ..units import Quantity
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


def parse_band(band):
    if isinstance(band, Band):
        return band
    if isinstance(band, Mapping):
        return Band(**band)
    if isinstance(band, str):
        return get_band(band)


def validate_band_config(band):
    if "passband" not in band:
        if any([key not in band for key in ["center", "width"]]):
            raise ValueError("The band's center and width must be specified")


def get_band(band_name):
    if band_name in BAND_CONFIGS:
        return Band(name=band_name, **BAND_CONFIGS[band_name])
    else:
        raise ValueError(f"'{band_name}' is not a valid pre-defined band name.")


def generate_passband(center, width, shape, samples=256):
    if shape == "flat":
        nu_min, nu_max = (center - 0.6 * width, center + 0.6 * width)
    elif shape == "top_hat":
        nu_min, nu_max = (center - width, center + width)
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
        raise ValueError(f"Invalid shape '{shape}'")

    if np.trapezoid(tau, x=nu) < 1e-2 * (nu_max - nu_min):
        raise ValueError("Error generating band")

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

        bad_freqs = list(self.nu[(self.nu < MARIA_MIN_NU) | (self.nu > MARIA_MAX_NU)])
        if bad_freqs:
            qmin_nu = Quantity(MARIA_MIN_NU, units="Hz")
            qmax_nu = Quantity(MARIA_MAX_NU, units="Hz")
            if nu is None:
                raise FrequencyOutOfBoundsError(center_and_width=(center, width))
            else:
                raise FrequencyOutOfBoundsError(nu=nu)

        # this turns e.g. 56MHz to "f056" and 150GHz to "f150"
        self.name = name or f"f{10 ** (np.log10(self.center) % 3):>03.0f}"
        self.shape = shape
        self.efficiency = efficiency
        self.NEP_per_loading = NEP_per_loading
        self.knee = knee
        self.time_constant = time_constant
        self.gain_error = gain_error

        self.spectrum_kwargs = {}
        if spectrum_kwargs:
            self.spectrum = AtmosphericSpectrum(region=spectrum_kwargs["region"])
            self.spectrum_kwargs["zenith_pwv"] = spectrum_kwargs.get("pwv", 1e0)
            self.spectrum_kwargs["base_temperature"] = spectrum_kwargs.get(
                "temperature", self.spectrum.side_base_temperature.mean()
            )
            self.spectrum_kwargs["elevation"] = np.radians(spectrum_kwargs.get("elevation", 45))
        else:
            self.spectrum = None

        if sensitivity:
            logger.warning(
                "The 'sensitivity' keyword is deprecated and will be removed in future releases. "
                "To specify noise levels in terms of sky temperature, use the 'NET_RJ' or 'NET_CMB' keywords instead."
            )
            NET_RJ = sensitivity

        if (NEP is None) and (NET_RJ is None) and (NET_CMB is None):
            logger.warning(f"No noise level specified for band {self.name}, assuming a sensitivity of 50 uK_RJ√s.")
            self.NET_RJ = 50e-6

        else:
            if NEP is not None:
                self.NEP = NEP
            elif NET_RJ is not None:
                self.NET_RJ = NET_RJ
            elif NET_CMB is not None:
                self.NET_CMB = NET_CMB

        self.transmission_integral_grids = {}

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
        filling = {
            "name": self.name,
            "center": Quantity(self.center, "Hz"),
            "width": Quantity(self.width, "Hz"),
            "η": self.efficiency,
            "NEP": Quantity(self.NEP, "W√s"),
            "NET_RJ": Quantity(self.NET_RJ, "K√s"),
            "NET_CMB": Quantity(self.NET_CMB, "K√s"),
        }

        summary = pd.Series(filling)
        return summary

    def __repr__(self):
        return f"Band({', '.join([f'{index}={entry}' for index, entry in self.summary().items()])})"

    #     def __repr__(self):
    #         return f"""{self.__class__.__name__}:
    #   name: {self.name}
    #   center: {Quantity(self.center, 'Hz')}
    #   width: {Quantity(self.width, 'Hz')}
    #   efficiency: {self.efficiency}
    #   NEP: {Quantity(self.NEP, "W√s")}
    #   NET_RJ: {Quantity(self.NET_RJ, "K√s")}
    #   NET_CMB: {Quantity(self.NET_CMB, "K√s")}
    # """
    def plot(self):
        fig, ax = plt.subplots(1, 1)

        qnu = Quantity(self.nu, "Hz")

        ax.plot(qnu.value, self.tau, label=self.name)

        ax.set_xlim(qnu.value.min(), qnu.value.max())
        ax.set_xlabel(rf"$\nu$ [${qnu.u['math_name']}$]")
        ax.set_ylabel(r"$\tau(\nu)$ [Rayleigh-Jeans]")
        ax.legend()

    @property
    def NET_RJ(self):
        return self.cal("W -> K_RJ", spectrum=self.spectrum, **self.spectrum_kwargs)(self.NEP).item()

    @NET_RJ.setter
    def NET_RJ(self, value):
        self.NEP = self.cal("K_RJ -> W", spectrum=self.spectrum, **self.spectrum_kwargs)(value).item()

    @property
    def NET_CMB(self):
        return self.cal("W -> K_CMB", spectrum=self.spectrum, **self.spectrum_kwargs)(self.NEP).item()

    @NET_CMB.setter
    def NET_CMB(self, value):
        self.NEP = self.cal("K_CMB -> W", spectrum=self.spectrum, **self.spectrum_kwargs)(value).item()

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
            return np.trapezoid(y=self.passband(nu), x=nu, axis=-1)

        else:
            nu = spectrum.side_nu[(spectrum.side_nu >= nu_min) & (spectrum.side_nu < nu_max)]
            integral_grid = np.trapezoid(y=self.passband(nu) * np.exp(-spectrum._opacity), x=nu, axis=-1)
            xi = (kwargs["base_temperature"], kwargs["zenith_pwv"], kwargs["elevation"])
            return np.array(jsp.interpolate.RegularGridInterpolator(points=spectrum.points[:3], values=integral_grid)(xi))

    def transmission(self, region="chajnantor", pwv=1, elevation=np.radians(90)) -> float:
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

    @property
    def wavelength(self):
        """
        Return the wavelength of the center, in meters.
        """
        return c / self.center

    def cal(self, signature: str, **kwargs) -> float:
        return Calibration(
            signature,
            band=self,
            **kwargs,
        )
