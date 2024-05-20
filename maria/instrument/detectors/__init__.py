import os

import matplotlib as mpl
import numpy as np
import pandas as pd

from ..bands import BandList
from .arrays import generate_array  # noqa

here, this_filename = os.path.split(__file__)

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

DET_COLUMN_TYPES = {
    "array": "str",
    "uid": "str",
    "band_name": "str",
    "band_center": "float",
    "sky_x": "float",
    "sky_y": "float",
    "baseline_x": "float",
    "baseline_y": "float",
    "baseline_z": "float",
    "pol_angle": "float",
    "pol_label": "str",
    "primary_size": "float",
    "bath_temp": "float",
    "time_constant": "float",
    "white_noise": "float",
    "pink_noise": "float",
    "efficiency": "float",
}

SUPPORTED_ARRAY_PACKINGS = ["hex", "square", "sunflower"]
SUPPORTED_ARRAY_SHAPES = ["hex", "square", "circle"]


class Detectors:
    def __repr__(self):
        return self.df.T.__repr__()

    def _repr_html_(self):
        return self.df.T._repr_html_()

    def __init__(self, df: pd.DataFrame, bands: dict = {}, config: dict = {}):
        self.df = df
        self.bands = bands
        self.config = config

        for band_attr, det_attr in {
            "center": "band_center",
            "width": "band_width",
        }.items():
            self.df.loc[:, det_attr] = getattr(self, band_attr)

    def __getattr__(self, attr):
        if attr in self.df.columns:
            return self.df.loc[:, attr].values.astype(DET_COLUMN_TYPES[attr])

        if all(hasattr(band, attr) for band in self.bands):
            values = np.zeros(shape=self.n, dtype=float)
            for band in self.bands:
                values[self.band_name == band.name] = getattr(band, attr)
            return values

        raise AttributeError(f"'Detectors' object has no attribute '{attr}'")

    def subset(self, band_name=None):
        bands = BandList([self.bands[band_name]])
        return Detectors(bands=bands, df=self.df.loc[self.band_name == band_name])

    @property
    def n(self):
        return len(self.df)

    @property
    def offsets(self):
        return np.c_[self.sky_x, self.sky_y]

    @property
    def __len__(self):
        return len(self.df)

    @property
    def index(self):
        return self.df.index.values

    @property
    def ubands(self):
        return list(self.bands.keys())

    def passband(self, nu):
        _nu = np.atleast_1d(nu)

        PB = np.zeros((len(self.df), len(_nu)))

        for band in self.bands:
            PB[self.band_name == band.name] = band.passband(_nu)

        return PB
