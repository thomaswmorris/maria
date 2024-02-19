import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

BAND_FIELD_TYPES = {
    "center": "float",
    "width": "float",
    "passband_shape": "str",
}

here, this_filename = os.path.split(__file__)


def generate_bands(bands_config):
    bands = []

    for band_key, band_config in bands_config.items():
        band_name = band_config.get("band_name", band_key)
        band_file = band_config.get("file")

        if band_file is not None:
            if os.path.exists(band_file):
                band_df = pd.read_csv(band_file)
            elif os.path.exists(f"{here}/{band_file}"):
                band_df = pd.read_csv(f"{here}/{band_file}")
            else:
                raise FileNotFoundError(band_file)

            band = Band.from_passband(
                name=band_name, nu=band_df.nu_GHz.values, pb=band_df.pb.values
            )

        else:
            band = Band(
                name=band_name,
                center=band_config["band_center"],
                width=band_config["band_width"],
                white_noise=band_config.get("white_noise", 0),
                pink_noise=band_config.get("pink_noise", 0),
                tau=band_config.get("tau", 0),
            )

        bands.append(band)

    return BandList(bands)


class BandList(Sequence):
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
        # return f"BandList([{', '.join(self.names)}])"
        return self.bands.__repr__()

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
    passband_shape: str = "flat"
    tau: float = 0
    white_noise: float = 0
    pink_noise: float = 0

    @classmethod
    def from_passband(cls, name, nu, pb, pb_err=None):
        center = np.round(np.sum(pb * nu), 3)
        width = np.round(
            nu[pb > pb.max() / np.e**2].ptp(), 3
        )  # width is the two-sigma interval

        band = cls(name=name, center=center, width=width, passband_shape="custom")

        band._nu = nu
        band._pb = pb

        return band

    @property
    def nu_min(self) -> float:
        if self.passband_shape == "flat":
            return self.center - 0.5 * self.width
        if self.passband_shape == "gaussian":
            return self.center - self.width
        if self.passband_shape == "custom":
            return self._nu[self._pb > 1e-2 * self._pb.max()].min()

    @property
    def nu_max(self) -> float:
        if self.passband_shape == "flat":
            return self.center + 0.5 * self.width
        if self.passband_shape == "gaussian":
            return self.center + self.width
        if self.passband_shape == "custom":
            return self._nu[self._pb > 1e-2 * self._pb.max()].max()

    def passband(self, nu):
        """
        Passband response as a function of nu (in GHz). These integrate to one.
        """

        _nu = np.atleast_1d(nu)

        if self.passband_shape == "gaussian":
            band_sigma = self.width / 4

            return np.exp(-0.5 * np.square((_nu - self.center) / band_sigma))

        if self.passband_shape == "flat":
            return np.where((_nu > self.nu_min) & (_nu < self.nu_max), 1.0, 0.0)

        elif self.passband_shape == "custom":
            return np.interp(_nu, self._nu, self._pb)
