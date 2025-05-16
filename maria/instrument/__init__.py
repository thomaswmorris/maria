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
from ..units import Quantity
from ..utils import HEX_CODE_LIST, flatten_config, get_rotation_matrix_2d, read_yaml  # noqa

here, this_filename = os.path.split(__file__)

INSTRUMENT_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    key = os.path.split(path)[1].split(".")[0]
    INSTRUMENT_CONFIGS[key] = read_yaml(path)
INSTRUMENT_CONFIGS = flatten_config(INSTRUMENT_CONFIGS)

# better formatting for pandas dataframes
# pd.set_eng_float_format()

for name, config in INSTRUMENT_CONFIGS.items():
    config["aliases"] = config.get("aliases", [])
    config["aliases"].append(name.lower())

INSTRUMENT_DISPLAY_COLUMNS = [
    "description",
    # "field_of_view",
    # "primary_size",
    # "bands",
]


# def get_instrument_config(instrument_name=None, **kwargs):
#     if instrument_name not in INSTRUMENT_CONFIGS.keys():
#         raise InvalidInstrumentError(instrument_name)
#     instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
#     return instrument_config


# def get_instrument(instrument_name="default", **kwargs):
#     """
#     Get an instrument from a pre-defined config.
#     """
#     if instrument_name:
#         for key, config in INSTRUMENT_CONFIGS.items():
#             if instrument_name.lower() in config.get("aliases", []):
#                 instrument_name = key
#         if instrument_name not in INSTRUMENT_CONFIGS.keys():
#             raise InvalidInstrumentError(instrument_name)
#         instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
#     else:
#         instrument_config = {}
#     instrument_config.update(kwargs)
#     return Instrument.from_config(instrument_config)


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


def get_instrument(name=None, **kwargs):
    config = get_instrument_config(name) if name else {}
    config.update(kwargs)
    return Instrument.from_config(config)


def get_instrument_config(name):
    for v in INSTRUMENT_CONFIGS.values():
        if name.lower() in v["aliases"]:
            return v.copy()
    raise KeyError(f"'{name}' is not a valid array name.")


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
        vel_limit: float = 1e2,  # in deg/s
        acc_limit: float = 1e2,  # in deg/s^2
    ):
        """
        Parameters
        ----------
        vel_limit : type
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
        self.vel_limit = vel_limit
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
    def sky_x(self):
        return self.dets.sky_x

    @property
    def sky_y(self):
        return self.dets.sky_y

    @property
    def offsets(self):
        return np.c_[self.sky_x, self.sky_y]

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
    def n_dets(self):
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

    def plot(self, z=np.inf, plot_pol_angles=True):
        self.dets.plot(z=z, plot_pol_angles=plot_pol_angles)


instrument_data = pd.DataFrame(INSTRUMENT_CONFIGS).reindex(INSTRUMENT_DISPLAY_COLUMNS).T

# for instrument_name, config in INSTRUMENT_CONFIGS.items():
#     instrument = get_instrument(instrument_name)
#     f_list = sorted(np.unique([band.center for band in instrument.dets.bands]))
#     instrument_data.at[instrument_name, "f [GHz]"] = "/".join([str(f) for f in f_list])
#     instrument_data.at[instrument_name, "n"] = instrument.dets.n

all_instruments = list(instrument_data.index)


class InvalidInstrumentError(Exception):
    def __init__(self, invalid_instrument):
        super().__init__(
            f"The instrument '{invalid_instrument}' is not supported. "
            f"Supported instruments are:\n\n{instrument_data.__repr__()}",
        )
