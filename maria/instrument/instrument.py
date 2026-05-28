from __future__ import annotations

import glob
import os
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from ..array import Array, ArrayList, get_array_config  # noqa
from ..band import BAND_CONFIGS, Band, BandList, parse_band  # noqa
from ..beam import compute_angular_fwhm
from ..io import flatten_config, read_yaml
from ..units import Quantity
from ..utils import HEX_CODE_LIST

subarray_params_to_inherit = [
    "array_packing",
    "array_shape",
    "array_offset",
    "beam_spacing",
    "max_baseline",
    "baseline_packing",
    "baseline_shape",
    "baseline_offset",
    "bath_temp",
    "polarization",
    "primary_size",
    "bands",
    "field_of_view",
]

band_params_to_inherit = [
    "time_constant",  # seconds
    "white_noise",  # Kelvin
    "pink_noise",  # Kelvin / s
    "efficiency",
]

passband_params_to_inherit = {
    "center": 150,  # GHz
    "width": 30,  # GHz
    "shape": "top_hat",
}

allowed_subarray_params = {
    "n": "int",
    "array_packing": "float",
    "array_shape": "float",
    "array_offset": "float",
    "beam_spacing": "float",
    "max_baseline": "float",
    "baseline_packing": "float",
    "baseline_shape": "float",
    "baseline_offset": "float",
    "bath_temp": "float",
    "polarization": "float",
    "primary_size": "float",
    "field_of_view": "float",
    "bands": "float",
}


class Instrument:
    @classmethod
    def from_config(cls, config):
        c = config.copy()

        if "array" in c:
            c["arrays"] = [c.pop("array")]

        for key in ["aliases"]:
            if key in c:
                c.pop(key)

        return cls(**c)

    def __init__(
        self,
        arrays: ArrayList | list | dict,
        description: str = "An instrument.",
        documentation: str = "",
        az_vel_limit: float = 1e2,  # in deg/s
        acc_limit: float = 1e2,  # in deg/s^2
    ):
        """
        Parameters
        ----------
        az_vel_limit : type
            The maximum angular speed of the array.
        """

        # if isinstance(arrays, list):
        #     array_index = 1
        #     for array in arrays:
        #         if "name" not in array:
        #             array["name"] = f"array{array_index}"
        #             array_index += 1

        # array_list = []
        # for array in arrays:
        #     if isinstance(array, Mapping):
        #         array_config = get_array_config(**array)
        #         array_list.append(Array.from_config(array_config))
        #     elif isinstance(array, Array):
        #         array_list.append(array)

        # if isinstance(arrays, ArrayList):
        #     self.arrays = arrays.arrays
        # else:
        #     if isinstance(arrays, list):
        #         array_names = [array.name for i in range(len(arrays))]
        #         array_values = arrays

        #     elif isinstance(arrays, dict):
        #         array_names = list(arrays.keys())
        #         array_values = list(arrays.values())
        #     else:
        #         raise ValueError("'arrays' must be a list or a dict.")

        #     self.arrays = []
        #     for array_name, array in zip(array_names, array_values):
        #         if isinstance(array, Array):
        #             array.name = array.name or array_name
        #             self.arrays.append(array)
        #         elif isinstance(array, dict):
        #             if "name" not in array:
        #                 array["name"] = array_name
        #             self.arrays.append(Array.from_config(array))
        #         elif isinstance(array, str):
        #             self.arrays.append(Array.from_kwargs(name=array_name, key=array))

        self.arrays = ArrayList(arrays)
        self.description = description
        self.documentation = documentation
        self.az_vel_limit = az_vel_limit
        self.acc_limit = acc_limit

        # self.primary_size = float(self.dets.primary_size.max())
        # self.field_of_view = np.round(np.degrees(lazy_diameter(self.dets.offsets)), 3)
        # self.baseline = np.round(lazy_diameter(self.dets.baselines), 3)

        # if self.field_of_view < 0.5 / 60:
        #     self.units = "arcsec"
        # elif self.field_of_view < 0.5:
        #     self.units = "arcmin"
        # else:
        #     self.units = "degrees"

    def __repr__(self):
        band_summary = self.arrays.bands.summary()
        band_summary.loc[:, "FWHM"] = [Quantity(self.dets(band=b.name).fwhm.mean(), "rad") for b in self.bands]

        arrays_repr = "\n".join([f"│  {s}" for s in str(self.arrays.summary().__repr__()).split("\n")])
        bands_repr = "\n".join([f"   {s}" for s in str(band_summary.__repr__()).split("\n")])

        s = f"""Instrument({len(self.arrays)} array{"s" if len(self.arrays) > 1 else ""})
├ arrays:
{arrays_repr}
│ 
└ bands:
{bands_repr}"""

        return s

    @property
    def dets(self):
        return self.arrays.combine()

    @property
    def bands(self):
        return self.arrays.bands

    @property
    def xi(self):
        return self.dets.xi

    @property
    def eta(self):
        return self.dets.eta

    @property
    def offsets(self):
        return np.c_[self.xi, self.eta]

    @property
    def baseline_x(self):
        return self.dets.baseline_x

    @property
    def baseline_y(self):
        return self.dets.baseline_y

    @property
    def baseline_z(self):
        return self.dets.baseline_z

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y, self.baseline_z]

    @staticmethod
    def beam_profile(r, fwhm):
        return np.exp(np.log(0.5) * np.abs(r / fwhm) ** 8)

    @property
    def n(self):
        return self.dets.n

    # def angular_beam_filter(self, z, res, beam_profile=None, buffer=1):  # noqa F401
    #     """
    #     Angular beam width (in radians) as a function of depth (in meters)
    #     """
    #     return construct_beam_filter(self.angular_fwhm(z), res, beam_profile=beam_profile, buffer=buffer)

    # def physical_beam_filter(self, z, res, beam_profile=None, buffer=1):  # noqa F401
    #     """
    #     Angular beam width (in radians) as a function of depth (in meters)
    #     """
    #     return construct_beam_filter(self.physical_fwhm(z), res, beam_profile=beam_profile, buffer=buffer)

    def plot(self, z=np.inf, plot_gammas=False):
        self.dets.plot(z=z, plot_gammas=plot_gammas)
