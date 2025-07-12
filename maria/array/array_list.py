from __future__ import annotations

import copy
import os
from typing import Mapping

import numpy as np
import pandas as pd

from ..band import BandList  # noqa
from ..beam import compute_angular_fwhm
from ..units import Quantity
from ..utils import HEX_CODE_LIST, compute_diameter
from .array import Array, get_array, get_array_config

here, this_filename = os.path.split(__file__)


class ArrayList:
    def __init__(self, arrays: list[Mapping]):
        if isinstance(arrays, Mapping):
            # convert to a list
            arrays = [{"name": k, **v} for k, v in arrays.items()]

        if isinstance(arrays, list):
            unnamed_array_index = 1
            self.arrays = []
            for array in arrays:
                if isinstance(array, str):
                    self.arrays.append(get_array(array))
                elif isinstance(array, Mapping):
                    if "name" not in array:
                        array["name"] = f"array{unnamed_array_index}"
                        unnamed_array_index += 1
                    array_config = get_array_config(**array)
                    self.arrays.append(Array.from_config(array_config))
                elif isinstance(array, Array):
                    self.arrays.append(array)
                else:
                    raise ValueError("Arrays must be either a string or a mapping")
        elif isinstance(arrays, ArrayList):
            self.arrays = arrays.arrays
        else:
            raise ValueError("Each element of 'arrays' must be either an Array, a string, or a mapping.")

    def combine(self):
        array_dets = []
        for array in self.arrays:
            df = copy.deepcopy(array.dets)
            df.loc[:, "array_name"] = array.name
            array_dets.append(df)
        return Array(name="", dets=pd.concat(array_dets), bands=self.bands)

    def one_detector_from_each_band(self):
        return ArrayList(arrays=[array.one_detector_from_each_band() for array in self.arrays])

    def outer(self):
        return ArrayList(arrays=[array.outer() for array in self.arrays])

    @property
    def field_of_view(self):
        return Quantity(compute_diameter(self.offsets), units="rad")

    @property
    def max_baseline(self):
        return Quantity(compute_diameter(self.baselines), units="m")

    @property
    def n(self):
        return sum([array.n for array in self.arrays])

    @property
    def dets(self):
        return pd.concat([array.dets for array in self.arrays])

    @property
    def bands(self):
        bands = []
        for array in self.arrays:
            for band in array.bands:
                if band not in bands:
                    bands.append(band)
        return BandList(bands)

    def angular_fwhm(self, z):  # noqa F401
        """
        Angular beam width (in radians) as a function of depth (in meters)
        """
        return compute_angular_fwhm(z=z, fwhm_0=self.primary_size, n=1, nu=self.band_center)

    def physical_fwhm(self, z):
        """
        Physical beam width (in meters) as a function of depth (in meters)
        """
        return z * self.angular_fwhm(z)

    def mask(self, **kwargs):
        return np.concatenate([array.mask(**kwargs) for array in self.arrays], axis=0)

    def subset(self, **kwargs):
        return ArrayList([array.subset(**kwargs).dets for array in self.arrays])

    def summary(self):
        return pd.concat([array.summary() for array in self.arrays], axis=1).T

    @property
    def array_name(self):
        return np.concatenate([array.n * [array.name] for array in self.arrays], axis=0)

    @property
    def offsets(self):
        return np.concatenate([array.offsets for array in self.arrays], axis=0)

    @property
    def baselines(self):
        return np.concatenate([array.baselines for array in self.arrays], axis=0)

    def passband(self, nu):
        return np.concatenate([array.passband(nu) for array in self.arrays], axis=0)

    def __getitem__(self, key):
        return self.arrays[key]

    def __getattr__(self, attr):
        try:
            return np.concatenate([getattr(array, attr) for array in self.arrays], axis=0)
        except Exception:
            pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __repr__(self):
        return self.summary().__repr__()

    def _repr_html_(self):
        return self.summary()._repr_html_()

    def __iter__(self):  # it has to be called this
        return iter(self.arrays)  # return the list's iterator

    def __len__(self):
        return len(self.arrays)
