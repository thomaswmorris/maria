import glob
import os
from collections.abc import Mapping
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from ...functions import planck_spectrum
from ...io import flatten_config, read_yaml
from ...spectrum import AtmosphericSpectrum
from ...units import parse_tod_units, prefixes
from ...units.constants import T_CMB, c, k_B

here, this_filename = os.path.split(__file__)

FIELD_FORMATS = pd.read_csv(f"{here}/format.csv", index_col=0)

BAND_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    BAND_CONFIGS[tag] = read_yaml(path)

BAND_CONFIGS = flatten_config(BAND_CONFIGS)


def validate_band_config(band):
    if "passband" not in band:
        if any([key not in band for key in ["center", "width"]]):
            raise ValueError("The band's center and width must be specified!")


def get_band(band_name):
    if band_name in BAND_CONFIGS:
        return Band(name=band_name, **BAND_CONFIGS[band_name])
    else:
        raise ValueError(f"'{band_name}' is not a valid pre-defined band name.")


def parse_bands(bands):
    """
    Take in a flexible format of a band specification, and return a list of bands.
    """
    band_list = []

    if isinstance(bands, list):
        for band in bands:
            if isinstance(band, Band):
                band_list.append(band)
            elif isinstance(band, str):
                band_list.append(get_band(band_name=band))
            else:
                raise TypeError("'band' must be either a Band or a string.")
        return band_list

    elif isinstance(bands, Mapping):
        for band_name, band in bands.items():
            if isinstance(band, Band):
                band_list.append(band)
            elif isinstance(band, Mapping):
                band_list.append(Band(name=band_name, **band))

    else:
        raise TypeError("'bands' must be either a list or a mapping.")

    return band_list


def parse_tod_calibration_signature(s):
    res = {}
    for sep in ["->"]:
        if s.count(sep) == 1:
            if sep is not None:
                items = [u.strip() for u in s.split(sep)]
                if len(items) == 2:
                    for io, u in zip(["in", "out"], items):
                        match = parse_tod_units(u)
                        prefix = match["prefix"]
                        res[f"{io}_factor"] = (
                            prefixes.loc[prefix].factor if prefix else 1e0
                        )
                        res[f"{io}_units"] = match["base"]
        return res
    raise ValueError("Calibration must have signature 'units1 -> units2'.")


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
        efficiency: float = 1.0,
        sensitivity: float = None,
        sensitivity_kind: str = "rayleigh-jeans",
        gain_error: float = 0,
        NEP: float = None,
        NEP_per_loading: float = 0.0,
        knee: float = 1.0,
        time_constant: float = 0.0,
    ):
        auto = center is not None and width is not None
        manual = nu is not None and tau is not None

        if not auto ^ manual:
            raise ValueError(
                "You must pass either both 'center' and 'width' or both 'nu' and 'tau'."
            )

        if auto:
            self.nu, self.tau = generate_passband(center, width, shape, samples=64)

        if manual:
            tau_max = np.max(tau)
            efficiency *= tau_max

            self.nu = np.array(nu)
            self.tau = np.array(tau) / tau_max

            if (
                (self.nu.ndim != 1)
                or (self.tau.ndim != 1)
                or (self.nu.shape != self.tau.shape)
            ):
                raise ValueError(
                    f"'nu' and 'tau' have mismatched shapes ({self.nu.shape} and {self.tau.shape})."
                )

        self._name = name
        self.shape = shape
        self.efficiency = efficiency
        self.NEP_per_loading = NEP_per_loading

        self.knee = knee
        self.time_constant = time_constant

        self.gain_error = gain_error

        if (NEP is None) and (sensitivity is None):
            self.sensitivity = 1e-6

        elif (NEP is not None) and (sensitivity is not None):
            raise RuntimeError(
                "When defining a band, you must specify exactly one of 'NEP' or 'sensitivity'."
            )  # noqa

        elif NEP is not None:
            self.NEP = NEP

        elif sensitivity is not None:
            self.set_sensitivity(
                sensitivity, kind=sensitivity_kind
            )  # this sets the NEP automatically

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
            [
                sp.interpolate.interp1d(self.tau[[i, i + 1]], self.nu[[i, i + 1]])(0.5)
                for i in crossovers
            ]
        )

    @property
    def summary(self):
        summary = pd.Series(index=FIELD_FORMATS.index, dtype=str)

        for field, entry in FIELD_FORMATS.iterrows():
            value = getattr(self, field)

            if entry["dtype"] == "float":
                if entry["format"] == "e":
                    s = f"{value:.02e}"
                else:
                    s = f"{value}"

                if entry.units != "none":
                    s = f"{s} {entry.units}"

            elif entry["dtype"] == "str":
                s = f"{value}"

            summary[field] = s

        return summary

    def __repr__(self):
        summary = self.summary
        parts = []
        for field, entry in FIELD_FORMATS.iterrows():
            value = summary[field]
            s = f"{field}='{value}'" if entry["dtype"] == str else f"{field}={value}"
            parts.append(s)

        return f"Band({', '.join(parts)})"

    def plot(self):
        fig, ax = plt.subplots(1, 1)

        ax.plot(self.nu, self.tau, label=self.name)

        ax.set_xlabel(r"$\nu$ [GHz]")
        ax.set_ylabel(r"$\tau(\nu)$ [Rayleigh-Jeans]")
        ax.legend()

    @property
    def sensitivity(self):
        if not hasattr(self, "_sensitivity"):
            self._sensitivity = self.NEP * self.transmission() / self.dP_dTRJ
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value):
        self.set_sensitivity(value)

    def set_sensitivity(
        self, value, kind="rayleigh-jeans", region="chajnantor", pwv=0, elevation=90
    ):
        transmission = self.transmission(region=region, pwv=pwv, elevation=elevation)

        if kind.lower() == "rayleigh-jeans":
            self.NEP = 1e12 * self.dP_dTRJ * value / transmission

        elif kind.lower() == "cmb":
            self.NEP = 1e12 * self.dP_dTCMB * value / transmission

        self._sensitivity = value
        self._sensitivity_kind = kind

    def transmission(self, region="chajnantor", pwv=1, elevation=90) -> float:
        if not hasattr(self, "spectrum"):
            self.spectrum = AtmosphericSpectrum(region=region)
        elif self.spectrum.region != region:
            self.spectrum = AtmosphericSpectrum(region=region)
        return self.spectrum.transmission(nu=self.center, pwv=pwv, elevation=elevation)

    @classmethod
    def from_file(cls, filename, **kwargs):
        path = (
            f"{here}/{filename}" if os.path.exists(f"{here}/{filename}") else filename
        )

        df = pd.read_csv(path, index_col=0)

        nu, tau = df.nu.values, df.passband.values

        # dummy values for center and width
        band = cls(nu=nu, tau=tau, shape="custom", **kwargs)
        band.nu = nu
        band.tau = tau

    def passband(self, nu):
        return self.efficiency * sp.interpolate.interp1d(
            self.nu, self.tau, bounds_error=False, fill_value=0
        )(nu)

    @property
    def dP_dTRJ(self) -> float:
        """
        In watts per kelvin Rayleigh-Jeans, assuming perfect transmission.
        """

        # nu = np.linspace(self.nu_min, self.nu_max, 256)

        # dI_dTRJ = rayleigh_jeans_spectrum(nu=1e9 * nu, T=1)  # this is the same as the derivative
        # dP_dTRJ = np.trapezoid(dI_dTRJ * self.passband(nu), 1e9 * nu)

        dP_dTRJ = k_B * np.trapezoid(self.passband(self.nu), 1e9 * self.nu)

        return self.efficiency * dP_dTRJ

    @property
    def dP_dTCMB(self) -> float:
        """
        In watts per kelvin CMB, assuming perfect transmission.
        """

        eps = 1e-3
        delta_T = np.array([-eps / 2, eps / 2])

        TRJ = (
            planck_spectrum(nu=1e9 * self.nu, T=T_CMB + delta_T[:, None])
            * c**2
            / (2 * k_B * (1e9 * self.nu) ** 2)
        )

        return (
            self.efficiency
            * k_B
            * np.diff(np.trapezoid(TRJ * self.passband(self.nu), 1e9 * self.nu))[0]
            / eps
        )

    def cal(self, signature: str) -> float:
        """
        Remember that:
        d(out units) / d(in units) = (d(out units) / d(pW)) / (d(in units) / d(pW))
        """

        res = parse_tod_calibration_signature(signature)

        if res["in_units"] == "K_RJ":
            d_in_d_W = 1 / self.dP_dTRJ
        elif res["in_units"] == "K_CMB":
            d_in_d_W = 1 / self.dP_dTCMB
        else:
            d_in_d_W = 1

        if res["out_units"] == "K_RJ":
            d_out_d_W = 1 / self.dP_dTRJ
        elif res["out_units"] == "K_CMB":
            d_out_d_W = 1 / self.dP_dTCMB
        else:
            d_out_d_W = 1

        overall_factor = res["in_factor"] / res["out_factor"]

        return overall_factor * d_out_d_W / d_in_d_W

    @property
    def wavelength(self):
        """
        Return the wavelength of the center, in meters.
        """
        return c / (1e9 * self.center)


class BandList(Sequence):
    def __init__(self, bands: Union[Mapping, list]):
        self.bands = []

        if isinstance(bands, BandList):
            self.bands = bands

        elif isinstance(bands, Mapping):
            for band_name, band_config in bands.items():
                self.bands.append(Band(name=band_name, **band_config))

        elif isinstance(bands, list):
            for band in bands:
                if isinstance(band, Band):
                    self.bands.append(band)
                else:
                    self.bands.append(get_band(band))

    # @classmethod
    # def from_list(cls, bands):
    #     band_list = []
    #     for band in bands:
    #         if isinstance(band, str):
    #             band_list.append(Band(name=band, **BAND_CONFIGS[band]))
    #         elif isinstance(band, Band):
    #             band_list.append(band)
    #         else:
    #             raise ValueError("'band' must be either a Band or a string.")
    #     return cls(band_list)

    # @classmethod
    # def from_config(cls, config):
    #     bands = []

    #     if isinstance(config, str):
    #         config = read_yaml(f"{here}/{config}")

    #     for name in config.keys():
    #         band_config = config[name]
    #         if "file" in band_config.keys():
    #             band_config = read_yaml(f'{here}/{band_config["file"]}')

    #         bands.append(Band(name=name, **band_config))
    #     return cls(bands=bands)

    def plot(self):
        for band in self.bands:
            fig, ax = plt.subplots(1, 1)
            ax.plot(band.nu, band.tau, label=band.name)

        ax.set_xlabel(r"$\nu$ [GHz]")
        ax.set_ylabel(r"$\tau(\nu)$ [Rayleigh-Jeans]")
        ax.legend()

    def add(self, band):
        if not isinstance(band, Band):
            raise ValueError("'band' must be a Band type.")
        if band.name in self.names:
            raise RuntimeError(f"There is already a band called '{band.name}'.")
        self.bands.append(band)

    def __getattr__(self, attr):
        if attr in self.names:
            return self.__getitem__(attr)
        if all([hasattr(band, attr) for band in self.bands]):
            return [getattr(band, attr) for band in self.bands]
        raise AttributeError(f"BandList object has no attribute named '{attr}'.")

    def __getitem__(self, index):
        if type(index) is int:
            return self.bands[index]
        elif type(index) is str:
            if index not in self.names:
                raise ValueError(f"BandList has no band named {index}.")
            return self.bands[self.names.index(index)]
        else:
            raise ValueError(
                f"Invalid index {index}. A bandList must be indexed by either an integer or a string."
            )

    def __len__(self):
        return len(self.bands)

    def __repr__(self):
        return self.summary.__repr__()

    def _repr_html_(self):
        return self.summary._repr_html_()

    def __short_repr__(self):
        return f"BandList([{', '.join(self.names)}])"

    @property
    def names(self):
        return [band.name for band in self.bands]

    @property
    def summary(self) -> pd.DataFrame:
        summary = pd.DataFrame(index=self.names)

        for band in self.bands:
            band_summary = band.summary
            for field, entry in FIELD_FORMATS.iterrows():
                summary.loc[band.name, field] = band_summary[field]

        return summary
