from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from ..array import Array, ArrayList  # noqa
from ..band import BAND_CONFIGS, Band, BandList, parse_bands  # noqa
from ..units import Angle
from ..utils import HEX_CODE_LIST, flatten_config, get_rotation_matrix_2d, read_yaml

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
    "baseline_diameter",
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
    "baseline_diameter": "float",
    "baseline_packing": "float",
    "baseline_shape": "float",
    "baseline_offset": "float",
    "bath_temp": "float",
    "polarization": "float",
    "primary_size": "float",
    "field_of_view": "float",
    "bands": "float",
}


# def check_subarray_format(subarray):
#     if isinstance(subarray.get("file"), str):
#         return True
#     if isinstance(subarray, Mapping):
#         return all(k in allowed_subarray_params for k in subarray)
#     return False


# def get_subarrays(instrument_config):
#     """
#     Make the subarrays!
#     """
#     config = copy.deepcopy(instrument_config)

#     if ("array" in config) and ("subarrays" not in config):
#         subarray = config.pop("array")
#         if check_subarray_format(subarray):
#             config["subarrays"] = {"array": subarray}
#         else:
#             raise ValueError(f"Invalid array configuration: {subarray}")

#     subarrays = {}

#     for subarray_name in config["subarrays"]:
#         subarray = config["subarrays"][subarray_name]

#         if "file" in subarray:  # it points to a file:
#             if not os.path.exists(subarray["file"]):
#                 subarray["file"] = f"{here}/ArrayList/arrays/{subarray['file']}"
#             df = pd.read_csv(subarray["file"], index_col=0)

#             if "bands" not in subarray:
#                 subarray["bands"] = {}

#             subarray["n"] = len(df)
#             for band_name in np.unique(df.band_name.values):
#                 subarray["bands"][band_name] = Band(
#                     name=band_name, **BAND_CONFIGS[band_name]
#                 )

#         elif ("n" not in subarray) and ("field_of_view" not in subarray):
#             raise ValueError(
#                 "You must specificy one of 'n' or 'field_of_view' to generate an array."
#             )

#         subarray["bands"] = BandList(parse_bands(subarray["bands"]))

#         for param in subarray_params_to_inherit:
#             if (param in config) and (param not in subarray):
#                 subarray[param] = config[param]

#         # for band_name, band_config in subarray["bands"].items():
#         #     for param in band_params_to_inherit:
#         #         if param in config:
#         #         band_config[param] = band_config.get(param, default_value)

#         #     if "passband" not in band_config:
#         #         for param, default_value in passband_params_to_inherit:
#         #             band_config[param] = band_config.get(param, default_value)

#         subarrays[subarray_name] = subarray

#     return subarrays


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

        self.dets = ArrayList(arrays).combine()
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
        arrays_repr = self.dets.__repr__().replace("\n", "\n  ")
        return f"""Instrument:
{arrays_repr})"""

    @property
    def bands(self):
        return self.dets.bands

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

    def plot(self, z=np.inf, plot_baseline="infer", plot_pol_angles=False):
        if plot_baseline == "infer":
            plot_baseline = self.dets.max_baseline > 0

        if plot_baseline:
            fig, (focal_ax, baseline_ax) = plt.subplots(1, 2, figsize=(8, 5), dpi=160, constrained_layout=True)
        else:
            fig, focal_ax = plt.subplots(1, 1, figsize=(4, 4), dpi=160, constrained_layout=True)

        i = 0
        legend_handles = []

        focal_plane = Angle(self.dets.offsets)
        resolution = Angle(self.dets.fwhm)

        for ia, array in enumerate(self.dets.split()):
            for ib, band in enumerate(array.bands):
                c = HEX_CODE_LIST[i % len(HEX_CODE_LIST)]

                band_array = array(band=band.name)

                fwhms = Angle(band_array.angular_fwhm(z=z))
                offsets = Angle(band_array.offsets)
                pol_angles = Angle(band_array.pol_angle)
                baselines = band_array.baselines

                if plot_pol_angles:
                    dx = np.c_[-np.ones(band_array.n), np.ones(band_array.n)] / 2
                    dy = np.zeros((band_array.n, 2))

                    R = get_rotation_matrix_2d(pol_angles.radians)
                    dl = np.moveaxis(R @ np.stack([dx, dy], axis=1), 0, 2)
                    P = Angle(offsets.radians.T[:, None] + fwhms.radians * dl)
                    focal_ax.plot(*getattr(P, focal_plane.units), c=c, lw=5e-1)

                focal_ax.add_collection(
                    EllipseCollection(
                        widths=getattr(fwhms, focal_plane.units),
                        heights=getattr(fwhms, focal_plane.units),
                        angles=0,
                        units="xy",
                        facecolors=c,
                        edgecolors="k",
                        lw=1e-1,
                        alpha=0.5,
                        offsets=getattr(offsets, focal_plane.units),
                        transOffset=focal_ax.transData,
                    ),
                )

                legend_handles.append(
                    Patch(
                        label=f"{band.name}, (n={band_array.n}, "
                        f"res={getattr(fwhms, resolution.units).mean():.01f}{resolution.symbol})",
                        color=c,
                    ),
                )

                focal_ax.scatter(
                    *getattr(offsets, focal_plane.units).T,
                    # label=band.name,
                    s=0,
                    color=c,
                )

                i += 1

                if plot_baseline:
                    baseline_ax.scatter(*baselines.T[:2])

        focal_ax.set_xlabel(rf"$\theta_x$ offset ({focal_plane.units})")
        focal_ax.set_ylabel(rf"$\theta_y$ offset ({focal_plane.units})")
        focal_ax.legend(handles=legend_handles, fontsize=8)

        xls, yls = focal_ax.get_xlim(), focal_ax.get_ylim()
        cen_x, cen_y = np.mean(xls), np.mean(yls)
        wid_x, wid_y = np.ptp(xls), np.ptp(yls)
        radius = 0.5 * np.maximum(wid_x, wid_y)

        margin = getattr(fwhms, focal_plane.units).max()

        focal_ax.set_xlim(cen_x - radius - margin, cen_x + radius + margin)
        focal_ax.set_ylim(cen_y - radius - margin, cen_y + radius + margin)


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
