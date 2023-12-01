import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from operator import attrgetter
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from . import utils

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.cm.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

# better formatting for pandas dataframes
# pd.set_eng_float_format()

here, this_filename = os.path.split(__file__)

ARRAY_CONFIGS = utils.io.read_yaml(f"{here}/configs/arrays.yml")
ARRAY_PARAMS = set()
for key, config in ARRAY_CONFIGS.items():
    ARRAY_PARAMS |= set(config.keys())

DISPLAY_COLUMNS = ["description", "field_of_view", "primary_size"]
supported_arrays_table = pd.DataFrame(ARRAY_CONFIGS).T
all_arrays = supported_arrays_table.index.values


class InvalidArrayError(Exception):
    def __init__(self, invalid_array):
        super().__init__(
            f"The array '{invalid_array}' is not supported."
            f"Supported arrays are:\n\n{supported_arrays_table.loc[:, DISPLAY_COLUMNS].to_string()}"
        )


def get_array_config(array_name, **kwargs):
    if array_name not in ARRAY_CONFIGS.keys():
        raise InvalidArrayError(array_name)
    ARRAY_CONFIG = ARRAY_CONFIGS[array_name].copy()
    for k, v in kwargs.items():
        ARRAY_CONFIG[k] = v
    return ARRAY_CONFIG


def get_array(array_name, **kwargs):
    """
    Get an array from a pre-defined config.
    """
    return Array.from_config(get_array_config(array_name, **kwargs))


REQUIRED_DET_CONFIG_KEYS = ["n", "band_center", "band_width"]


def generate_dets_from_config(
    bands: Mapping,
    field_of_view: float,
    geometry: str = "hex",
    max_baseline: float = 0,
    randomize_offsets: bool = True,
):
    dets = pd.DataFrame(
        columns=[
            "band",
            "band_center",
            "band_width",
            "scan_center" "offset_x",
            "offset_y",
            "baseline_x",
            "baseline_y",
        ],
        dtype=float,
    )

    for band, band_config in bands.items():
        if not all(key in band_config.keys() for key in REQUIRED_DET_CONFIG_KEYS):
            raise ValueError(f"Each band must have keys {REQUIRED_DET_CONFIG_KEYS}")

        band_dets = pd.DataFrame(index=np.arange(band_config["n"]))
        band_dets.loc[:, "band"] = band
        band_dets.loc[:, "band_center"] = band_config["band_center"]
        band_dets.loc[:, "band_width"] = band_config["band_width"]

        dets = pd.concat([dets, band_dets])
        dets.index = np.arange(len(dets))

    # want cetner = scan_center similar to what is done in pointing.py
    offsets_radians = np.radians(
        utils.generate_array_offsets(geometry, field_of_view, len(dets), center=None)
    )

    # should we make another function for this?
    baseline = utils.generate_array_offsets(geometry, max_baseline, len(dets))

    if randomize_offsets:
        np.random.shuffle(offsets_radians)  # this is a stupid function.

    dets.loc[:, "offset_x"] = offsets_radians[:, 0]
    dets.loc[:, "offset_y"] = offsets_radians[:, 1]
    dets.loc[:, "baseline_x"] = baseline[:, 0]
    dets.loc[:, "baseline_y"] = baseline[:, 1]

    for key in ["offset_x", "offset_y", "baseline_x", "baseline_y"]:
        dets.loc[:, key] = dets.loc[:, key].astype(float)

    return dets


@dataclass
class Array:
    """
    An array.
    """

    description: str = ""
    primary_size: float = 5  # in meters
    field_of_view: float = 1  # in deg
    geometry: str = "hex"
    max_az_vel: float = 0  # in deg/s
    max_el_vel: float = np.inf  # in deg/s
    max_az_acc: float = 0  # in deg/s^2
    max_el_acc: float = np.inf  # in deg/s^2
    az_bounds: Tuple[float, float] = (0, 360)  # in degrees
    el_bounds: Tuple[float, float] = (0, 90)  # in degrees
    documentation: str = ""
    dets: pd.DataFrame = None  # dets, it's complicated

    def __repr__(self):
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self)) for f in fields(self) if f.name != "dets"
        )

        nodef_f_repr = ", ".join(f"{name}={value}" for name, value in nodef_f_vals)
        return f"{self.__class__.__name__}({nodef_f_repr})"

    @property
    def ubands(self):
        return np.unique(self.dets.band)

    @property
    def offset_x(self):
        return self.dets.offset_x.values

    @property
    def offset_y(self):
        return self.dets.offset_y.values

    @property
    def offsets(self):
        return np.c_[self.offset_x, self.offset_y]

    @property
    def baseline_x(self):
        return self.dets.baseline_x.values

    @property
    def baseline_y(self):
        return self.dets.baseline_y.values

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y]

    @classmethod
    def from_config(cls, config):
        if isinstance(config["dets"], Mapping):
            field_of_view = config.get("field_of_view", 1)
            geometry = config.get("geometry", "hex")
            dets = generate_dets_from_config(
                config["dets"], field_of_view=field_of_view, geometry=geometry
            )

        return cls(
            description=config["description"],
            primary_size=config["primary_size"],
            field_of_view=field_of_view,
            geometry=geometry,
            max_az_vel=config["max_az_vel"],
            max_el_vel=config["max_el_vel"],
            max_az_acc=config["max_az_acc"],
            max_el_acc=config["max_el_acc"],
            az_bounds=config["az_bounds"],
            el_bounds=config["el_bounds"],
            documentation=config["documentation"],
            dets=dets,
        )

    @staticmethod
    def beam_profile(r, fwhm):
        return np.exp(np.log(0.5) * np.abs(r / fwhm) ** 8)

    @property
    def n_dets(self):
        return len(self.dets)

    @property
    def band_min(self):
        return (self.dets.band_center - 0.5 * self.dets.band_width).values

    @property
    def band_max(self):
        return (self.dets.band_center + 0.5 * self.dets.band_width).values

    def passbands(self, nu):
        """
        Passband response as a function of nu (in GHz)
        """
        _nu = np.atleast_1d(nu)
        nu_mask = (_nu[None] > self.band_min[:, None]) & (
            _nu[None] < self.band_max[:, None]
        )
        return nu_mask.astype(float) / nu_mask.sum(axis=-1)[:, None]

    def angular_fwhm(self, z):
        return utils.gaussian_beam_angular_fwhm(
            z=z,
            w_0=self.primary_size / np.sqrt(2 * np.log(2)),
            f=self.dets.band_center.values,
            n=1,
        )

    def physical_fwhm(self, z):
        return z * self.angular_fwhm(z)

    def angular_beam(self, r, z=np.inf, n=1, l=None, f=None):  # noqa F401
        """
        Beam response as a function of radius (in radians)
        """
        return self.beam_profile(r, self.angular_fwhm(z))

    def physical_beam(self, r, z=np.inf, n=1, l=None, f=None):  # noqa F401
        """
        Beam response as a function of radius (in meters)
        """
        return self.beam_profile(r, self.physical_fwhm(z))

    def plot_dets(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=256)

        legend_handles = []
        for iub, uband in enumerate(self.ubands):
            band_color = HEX_CODE_LIST[iub]
            band_mask = self.dets.band.values == uband

            nom_freq = self.dets.band_center[band_mask].mean()
            band_res_arcmins = 60 * np.degrees(
                1.22 * 2.998e8 / (1e9 * nom_freq * self.primary_size)
            )
            # band_res_arcmins = 60 * np.degrees(utils.gaussian_beam_angular_fwhm(z=1e12, w_0=self.primary_size, f=nom_freq))
            offsets_arcmins = 60 * np.degrees(self.offsets[band_mask])

            ax.add_collection(
                EllipseCollection(
                    widths=band_res_arcmins,
                    heights=band_res_arcmins,
                    angles=0,
                    units="xy",
                    facecolors=band_color,
                    edgecolors="k",
                    lw=1e-1,
                    alpha=0.5,
                    offsets=offsets_arcmins,
                    transOffset=ax.transData,
                )
            )

            legend_handles.append(
                Patch(
                    label=f"{uband}, res = {band_res_arcmins:.01e} arcmin.",
                    color=band_color,
                )
            )

            ax.scatter(*offsets_arcmins.T, label=uband, s=5e-1, color=band_color)

        ax.set_xlabel(r"$\theta_x$ offset (arc min.)")
        ax.set_ylabel(r"$\theta_y$ offset (arc min.)")
        ax.legend(handles=legend_handles)
