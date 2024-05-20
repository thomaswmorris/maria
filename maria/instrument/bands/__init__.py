import copy
import glob
import os
from collections.abc import Mapping
from typing import Sequence

import numpy as np
import pandas as pd

from ...atmosphere.spectrum import Spectrum
from ...constants import T_CMB, c
from ...functions import planck_spectrum, rayleigh_jeans_spectrum
from ...io import flatten_config, read_yaml

BAND_FIELDS = {
    "center": {"units": "GHz", "dtype": "float"},
    "width": {"units": "GHz", "dtype": "float"},
    "shape": {"units": None, "dtype": "str"},
    "efficiency": {"units": None, "dtype": "float"},
    "sensitivity": {"units": "K sqrt(s)", "dtype": "float"},
    "NEP": {"units": "pW sqrt(s)", "dtype": "float"},
}

here, this_filename = os.path.split(__file__)

all_bands = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    all_bands[tag] = read_yaml(path)
all_bands = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    all_bands[tag] = read_yaml(path)

all_bands = flatten_config(all_bands)


def validate_band_config(band):
    if "passband" not in band:
        if any([key not in band for key in ["center", "width"]]):
            raise ValueError("The band's center and width must be specified!")


def parse_bands(bands):
    """
    There are many ways to specify bands, and this handles all of them.
    """
    bands = copy.deepcopy(bands)

    if isinstance(bands, Mapping):
        for name, band in bands.items():
            validate_band_config(band)

    # if we get a list of bands, convert them to a labeled dict
    if isinstance(bands, list):
        bands_mapping = {}
        for band in bands:
            if isinstance(band, str):
                # here 'band' is a name
                if band not in all_bands:
                    raise ValueError(f"Could not find band '{band}'.")
                bands_mapping[band] = all_bands[band]

            if isinstance(band, Mapping):
                if "center" not in band:
                    raise RuntimeError("You must specify the band center.")
                name = band.get("name", f'f{int(band["center"]):>03}')
                bands_mapping[name] = band

        return parse_bands(bands_mapping)

    return bands


class Band:
    def __init__(
        self,
        name: str,
        center: float,
        width: float,
        shape: str = "gaussian",
        efficiency: float = 1.0,
        sensitivity: float = None,
        sensitivity_kind: str = "rayleigh-jeans",
        NEP: float = None,
        NEP_per_loading: float = 0.0,
        knee: float = 1.0,
        time_constant: float = 0.0,
    ):
        self.name = name
        self.center = center
        self.width = width
        self.shape = shape
        self.efficiency = efficiency
        self.NEP_per_loading = NEP_per_loading

        self.knee = knee
        self.time_constant = time_constant

        self.spectrum = Spectrum(region="chajnantor")

        if NEP is not None:
            if sensitivity is not None:
                raise RuntimeError(
                    "When defining a band, you must specify exactly one of 'NEP' or 'sensitivity'."
                )
            self.NEP = NEP

        if sensitivity is not None:
            self.set_sensitivity(
                sensitivity, kind=sensitivity_kind
            )  # this sets the NEP automatically

    def __repr__(self):
        parts = []
        for field, d in BAND_FIELDS.items():
            value = getattr(self, field)
            if d["dtype"] == "str":
                s = f"{field}='{value}'"
            elif d["units"] is not None:
                s = f"{field}={value} {d['units']}"
            else:
                s = f"{field}={value}"
            parts.append(s)

        return f"Band({', '.join(parts)})"

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
        transmission = self.transmission(
            region=region, zenith_pwv=pwv, elevation=elevation
        )

        if kind.lower() == "rayleigh-jeans":
            self.NEP = self.dP_dTRJ * value / transmission

        elif kind.lower() == "cmb":
            self.NEP = self.dP_dTCMB * value / transmission

        self._sensitivity = value
        self._sensitivity_kind = kind

    def transmission(self, region="chajnantor", zenith_pwv=1, elevation=90) -> float:
        if self.spectrum.region != region:
            self.spectrum = Spectrum(region=region)
        return self.spectrum.transmission(
            nu=self.center, zenith_pwv=zenith_pwv, elevation=elevation
        )

    @classmethod
    def from_config(cls, name, config):
        if "passband" in config:
            df = pd.read_csv(f"{here}/{config.pop('passband')}", index_col=0)

            nu, pb = df.nu.values, df.passband.values

            center = np.round(np.sum(pb * nu), 3)
            width = np.round(nu[pb > 1e-2 * pb.max()].ptp(), 3)

            band = cls(name=name, center=center, width=width, shape="custom", **config)
            band._nu = nu
            band._pb = pb
        else:
            band = cls(name=name, **config)

        return band

    @property
    def nu_min(self) -> float:
        if self.shape == "flat":
            return self.center - 0.5 * self.width
        if self.shape == "custom":
            return self._nu[self._pb > 1e-2 * self._pb.max()].min()

        return self.center - self.width

    @property
    def nu_max(self) -> float:
        if self.shape == "flat":
            return self.center + 0.5 * self.width
        if self.shape == "custom":
            return self._nu[self._pb > 1e-2 * self._pb.max()].max()

        return self.center + self.width

    def passband(self, nu):
        """
        Passband response as a function of nu (in GHz).
        """
        _nu = np.atleast_1d(nu)

        if self.shape == "top_hat":
            return np.exp(np.log(0.5) * (2 * (_nu - self.center) / self.width) ** 8)

        if self.shape == "gaussian":
            return np.exp(np.log(0.5) * (2 * (_nu - self.center) / self.width) ** 2)

        if self.shape == "flat":
            return np.where((_nu > self.nu_min) & (_nu < self.nu_max), 1.0, 0.0)

        elif self.shape == "custom":
            return np.interp(_nu, self._nu, self._pb)

    @property
    def dP_dTRJ(self) -> float:
        """
        In watts per kelvin Rayleigh-Jeans, assuming perfect transmission.
        """

        nu = np.linspace(self.nu_min, self.nu_max, 256)

        dI_dTRJ = rayleigh_jeans_spectrum(
            nu=1e9 * nu, T=1
        )  # this is the same as the derivative
        dP_dTRJ = np.trapz(dI_dTRJ * self.passband(nu), 1e9 * nu)

        return self.efficiency * dP_dTRJ

    @property
    def dP_dTCMB(self) -> float:
        """
        In watts per kelvin CMB, assuming perfect transmission.
        """

        eps = 1e-3

        nu = np.linspace(self.nu_min, self.nu_max, 256)

        delta_T = np.array([-eps / 2, eps / 2])
        dI_dTCMB = (
            np.diff(planck_spectrum(nu=1e9 * nu, T=T_CMB + delta_T[:, None]), axis=0)[0]
            / eps
        )
        dP_dTCMB = self.efficiency * np.trapz(dI_dTCMB * self.passband(nu), 1e9 * nu)

        return self.efficiency * dP_dTCMB

    @property
    def wavelength(self):
        """
        Return the wavelength of the center, in meters.
        """
        return c / (1e9 * self.center)


class BandList(Sequence):
    @classmethod
    def from_config(cls, config):
        bands = []

        if isinstance(config, str):
            config = read_yaml(f"{here}/{config}")

        for name in config.keys():
            band_config = config[name]
            if "file" in band_config.keys():
                band_config = read_yaml(f'{here}/{band_config["file"]}')

            bands.append(Band(name=name, **band_config))
        return cls(bands=bands)

    def add(self, band):
        if not isinstance(band, Band):
            raise ValueError("'band' must be a Band type.")
        if band.name in self.names:
            raise RuntimeError(f"There is already a band called '{band.name}'.")
        self.bands.append(band)

    def __init__(self, bands: list = []):
        self.bands = bands

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
        table = pd.DataFrame(columns=list(BAND_FIELDS.keys()), index=self.names)

        for attr, d in BAND_FIELDS.items():
            dtype = d["dtype"]
            for band in self.bands:
                table.at[band.name, attr] = getattr(band, attr)
            table[attr] = table[attr].astype(dtype)

        return table
