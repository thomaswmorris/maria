import glob
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .. import utils
from ..constants import k_B
from ..utils.io import flatten_config, read_yaml

BAND_FIELD_TYPES = {
    "center": "float",
    "width": "float",
    "shape": "str",
    "time_constant": "float",
    "white_noise": "float",
    "pink_noise": "float",
    "efficiency": "float",
}

here, this_filename = os.path.split(__file__)

all_bands = {}
for path in glob.glob(f"{here}/bands/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    all_bands[tag] = read_yaml(path)
all_bands = {}
for path in glob.glob(f"{here}/bands/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    all_bands[tag] = read_yaml(path)

all_bands = flatten_config(all_bands)


class BandList(Sequence):
    @classmethod
    def from_config(cls, config):
        bands = []

        if isinstance(config, str):
            config = utils.io.read_yaml(f"{here}/{config}")

        for name in config.keys():
            band_config = config[name]
            if "file" in band_config.keys():
                band_config = utils.io.read_yaml(f'{here}/{band_config["file"]}')

            bands.append(Band(name=name, **band_config))
        return cls(bands=bands)

    def __init__(self, bands: list = []):
        self.bands = bands

    def __getattr__(self, attr):
        if attr in self.names:
            return self.__getitem__(attr)
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

    def __repr_html__(self):
        return self.summary.__repr_html__()

    def __short_repr__(self):
        return f"BandList([{', '.join(self.names)}])"

    @property
    def names(self):
        return [band.name for band in self.bands]

    @property
    def summary(self) -> pd.DataFrame:
        table = pd.DataFrame(columns=list(BAND_FIELD_TYPES.keys()), index=self.names)

        for attr, dtype in BAND_FIELD_TYPES.items():
            for band in self.bands:
                table.at[band.name, attr] = getattr(band, attr)
            table[attr] = table[attr].astype(dtype)

        return table


@dataclass
class Band:
    name: str
    center: float
    width: float
    time_constant: float = 0.0
    white_noise: float = 0.0
    pink_noise: float = 0.0
    shape: str = "top_hat"
    efficiency: float = 1.0

    @classmethod
    def from_passband(cls, name, nu, pb, pb_err=None):
        center = np.round(np.sum(pb * nu), 3)
        width = np.round(nu[pb > 1e-2 * pb.max()].ptp(), 3)

        band = cls(name=name, center=center, width=width, shape="custom")

        band._nu = nu
        band._pb = pb

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
            return np.exp(-np.log(2) * (2 * (_nu - self.center) / self.width) ** 8)

        if self.shape == "gaussian":
            return np.exp(-np.log(2) * (2 * (_nu - self.center) / self.width) ** 2)

        if self.shape == "flat":
            return np.where((_nu > self.nu_min) & (_nu < self.nu_max), 1.0, 0.0)

        elif self.shape == "custom":
            return np.interp(_nu, self._nu, self._pb)

    @property
    def abs_cal_rj(self):
        """
        Absolute calibration (i.e. temperature per power) in Kelvins per Watt.
        """
        nu = np.linspace(self.nu_min, self.nu_max, 256)
        dP_dT = self.efficiency * np.trapz(self.passband(nu) * k_B, 1e9 * nu)
        return 1 / dP_dT
