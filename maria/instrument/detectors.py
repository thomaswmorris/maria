import os

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy as sp

from ..units import Angle
from ..utils import compute_diameter
from .band import BandList
from .beam import compute_angular_fwhm

here, this_filename = os.path.split(__file__)

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

DET_COLUMN_TYPES = {
    "array": "str",
    "uid": "str",
    "array_name": "str",
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
    def __init__(self, df: pd.DataFrame, bands: BandList, config: dict = {}):
        self.df = df
        self.df.index = np.arange(len(self.df.index))

        self.bands = BandList(bands)
        self.config = config

        for band_attr, det_attr in {
            "center": "band_center",
            "width": "band_width",
        }.items():
            self.df.loc[:, det_attr] = getattr(self, band_attr)

    def mask(self, **kwargs):
        mask = np.ones(len(self.df)).astype(bool)
        for k, v in kwargs.items():
            mask &= self.df.loc[:, k].values == v
        return mask

    def subset(self, **kwargs):
        return self._subset(self.mask(**kwargs))

    def _subset(self, mask):
        df = self.df.loc[mask]
        return Detectors(
            df=df, bands=[b for b in self.bands if b.name in self.df.band_name.values]
        )

    def one_detector_from_each_band(self):
        first_det_mask = np.isin(
            np.arange(self.n), np.unique(self.band_name, return_index=True)[1]
        )
        return self._subset(mask=first_det_mask)

    def outer(self):
        outer_dets_index = sp.spatial.ConvexHull(self.offsets).vertices
        outer_dets_mask = np.isin(np.arange(self.n), outer_dets_index)
        return self._subset(mask=outer_dets_mask)

    @property
    def n(self):
        return len(self.df)

    @property
    def offsets(self):
        return np.c_[self.sky_x, self.sky_y]

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y, self.baseline_z]

    @property
    def field_of_view(self):
        return Angle(compute_diameter(self.offsets))

    @property
    def max_baseline(self):
        return compute_diameter(self.baselines)

    @property
    def index(self):
        return self.df.index.values

    @property
    def ubands(self):
        return list(self.bands.keys())

    @property
    def fwhm(self):
        """
        Returns the angular FWHM (in radians) at infinite distance.
        """
        return self.angular_fwhm(z=np.inf)

    def angular_fwhm(self, z):  # noqa F401
        """
        Angular beam width (in radians) as a function of depth (in meters)
        """
        nu = self.band_center  # in GHz
        return compute_angular_fwhm(z=z, fwhm_0=self.primary_size, n=1, nu=nu)

    def physical_fwhm(self, z):
        """
        Physical beam width (in meters) as a function of depth (in meters)
        """
        return z * self.angular_fwhm(z)

    def passband(self, nu):
        _nu = np.atleast_1d(nu)
        PB = np.zeros((len(self.df), len(_nu)))
        for band in self.bands:
            PB[self.band_name == band.name] = band.passband(_nu)
        return PB

    def cal(self, signature: str) -> float:
        """ """
        c = np.zeros(self.n)
        for band in self.bands:
            c[self.mask(band_name=band.name)] = band.cal(signature)
        return c

    def __getattr__(self, attr):
        if attr in self.df.columns:
            return self.df.loc[:, attr].values.astype(DET_COLUMN_TYPES[attr])

        if all(hasattr(band, attr) for band in self.bands):
            values = np.zeros(shape=self.n, dtype=float)
            for band in self.bands:
                values[self.band_name == band.name] = getattr(band, attr)
            return values

        raise AttributeError(f"'Detectors' object has no attribute '{attr}'")

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()
