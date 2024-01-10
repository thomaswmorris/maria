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
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

# better formatting for pandas dataframes
# pd.set_eng_float_format()

here, this_filename = os.path.split(__file__)

all_array_params = utils.io.read_yaml(f"{here}/configs/default_params.yml")["array"]

ARRAY_CONFIGS = utils.io.read_yaml(f"{here}/configs/arrays.yml")

DISPLAY_COLUMNS = ["array_description", "field_of_view", "primary_size", "bands"]
array_data = pd.DataFrame(ARRAY_CONFIGS).T

band_lists = []
for array_name, config in ARRAY_CONFIGS.items():
    band_lists.append(
        "/".join(
            [
                str(band_config["band_center"])
                for band_config in config["detector_config"].values()
            ]
        )
    )
array_data.loc[:, "bands"] = band_lists

all_arrays = list(array_data.index.values)


class InvalidArrayError(Exception):
    def __init__(self, invalid_array):
        super().__init__(
            f"The array '{invalid_array}' is not supported."
            f"Supported arrays are:\n\n{array_data.loc[:, DISPLAY_COLUMNS].__repr__()}"
        )


def generate_array_offsets(geometry, field_of_view, n):
    valid_array_types = ["flower", "hex", "square"]

    if geometry == "flower":
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        dzs = np.zeros(n).astype(complex)
        for i in range(n):
            dzs[i] = np.sqrt((i / (n - 1)) * 2) * np.exp(1j * phi * i)
        od = np.abs(np.subtract.outer(dzs, dzs))
        dzs *= field_of_view / od.max()
        return np.c_[np.real(dzs), np.imag(dzs)]
    if geometry == "hex":
        return generate_hex_offsets(n, field_of_view)
    if geometry == "square":
        dxy_ = np.linspace(-field_of_view, field_of_view, int(np.ceil(np.sqrt(n)))) / (
            2 * np.sqrt(2)
        )
        DX, DY = np.meshgrid(dxy_, dxy_)
        return np.c_[DX.ravel()[:n], DY.ravel()[:n]]

    raise ValueError(
        "Please specify a valid array type. Valid array types are:\n"
        + "\n".join(valid_array_types)
    )


def generate_hex_offsets(n, d):
    angles = np.linspace(0, 2 * np.pi, 6 + 1)[1:] + np.pi / 2
    zs = np.array([0])
    layer = 0
    while len(zs) < n:
        for angle in angles:
            for z in layer * np.exp(1j * angle) + np.arange(layer) * np.exp(
                1j * (angle + 2 * np.pi / 3)
            ):
                zs = np.append(zs, z)
        layer += 1
    zs -= zs.mean()
    zs *= 0.5 * d / np.abs(zs).max()

    return np.c_[np.real(np.array(zs[:n])), np.imag(np.array(zs[:n]))]


def get_array_config(array_name=None, **kwargs):
    if array_name not in ARRAY_CONFIGS.keys():
        raise InvalidArrayError(array_name)
    array_config = ARRAY_CONFIGS[array_name].copy()
    for key, value in kwargs.items():
        if key in all_array_params.keys():
            array_config[key] = value
        else:
            raise ValueError(f"'{key}' is not a valid argument for an array!")
    return array_config


def get_array(array_name="default", **kwargs):
    """
    Get an array from a pre-defined config.
    """
    array_config = get_array_config(array_name=array_name, **kwargs)
    return Array.from_config(array_config)


REQUIRED_DET_CONFIG_KEYS = ["n", "band_center", "band_width"]


def generate_dets_from_config(
    bands: Mapping,
    field_of_view: float,
    geometry: str = "hex",
    baseline: float = 0,
    randomize_offsets: bool = True,
):
    dets = pd.DataFrame(
        columns=[
            "band",
            "band_center",
            "band_width",
            "offset_x",
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

        det_offsets_radians = np.radians(
            generate_array_offsets(geometry, field_of_view, len(band_dets))
        )

        # should we make another function for this?
        det_baselines_meters = generate_array_offsets(
            geometry, baseline, len(band_dets)
        )

        # if randomize_offsets:
        #     np.random.shuffle(offsets_radians)  # this is a stupid function.

        band_dets.loc[:, "offset_x"] = det_offsets_radians[:, 0]
        band_dets.loc[:, "offset_y"] = det_offsets_radians[:, 1]
        band_dets.loc[:, "baseline_x"] = det_baselines_meters[:, 0]
        band_dets.loc[:, "baseline_y"] = det_baselines_meters[:, 1]
        band_dets.loc[:, "baseline_z"] = 0

        dets = pd.concat([dets, band_dets])
        dets.index = np.arange(len(dets))

    for key in ["offset_x", "offset_y", "baseline_x", "baseline_y"]:
        dets.loc[:, key] = dets.loc[:, key].astype(float)

    return dets


@dataclass
class Array:
    """
    An array.
    """

    array_description: str = ""
    primary_size: float = 5  # in meters
    field_of_view: float = 1  # in deg
    geometry: str = "hex"
    baseline: float = 0
    max_az_vel: float = 0  # in deg/s
    max_el_vel: float = np.inf  # in deg/s
    max_az_acc: float = 0  # in deg/s^2
    max_el_acc: float = np.inf  # in deg/s^2
    az_bounds: Tuple[float, float] = (0, 360)  # in degrees
    el_bounds: Tuple[float, float] = (0, 90)  # in degrees
    dets: pd.DataFrame = None  # dets, it's complicated
    array_documentation: str = ""

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
        if isinstance(config["detector_config"], Mapping):
            field_of_view = config.get("field_of_view", 1)
            geometry = config.get("geometry", "hex")
            baseline = config.get("baseline", 0)  # default to zero baseline
            dets = generate_dets_from_config(
                bands=config["detector_config"],
                field_of_view=field_of_view,
                geometry=geometry,
                baseline=baseline,
            )

        return cls(
            array_description=config["array_description"],
            primary_size=config["primary_size"],
            field_of_view=field_of_view,
            baseline=baseline,
            geometry=geometry,
            max_az_vel=config["max_az_vel"],
            max_el_vel=config["max_el_vel"],
            max_az_acc=config["max_az_acc"],
            max_el_acc=config["max_el_acc"],
            az_bounds=tuple(config["az_bounds"]),
            el_bounds=tuple(config["el_bounds"]),
            array_documentation=config["array_documentation"],
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

    @property
    def fwhm(self):
        """
        Returns the angular FWHM (in radians) at infinite distance.
        """
        nu = self.dets.band_center.values  # in GHz
        return utils.beam.angular_fwhm(
            z=np.inf, fwhm_0=self.primary_size, n=1, f=1e9 * nu
        )

    def physical_fwhm(self, z):
        """
        Physical beam width (in meters) as a function of depth (in meters)
        """
        return z * self.angular_fwhm(z)

    def angular_fwhm(self, z):  # noqa F401
        """
        Angular beam width (in radians) as a function of depth (in meters)
        """
        nu = self.dets.band_center.values  # in GHz
        return utils.beam.angular_fwhm(z=z, fwhm_0=self.primary_size, n=1, f=1e9 * nu)

    def passbands(self, nu):
        """
        Passband response as a function of nu (in GHz)
        """
        _nu = np.atleast_1d(nu)
        nu_mask = (_nu[None] > self.band_min[:, None]) & (
            _nu[None] < self.band_max[:, None]
        )
        return nu_mask.astype(float) / nu_mask.sum(axis=-1)[:, None]

    def plot_dets(self, units="deg"):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=256)

        legend_handles = []
        for iub, uband in enumerate(self.ubands):
            band_mask = self.dets.band.values == uband

            fwhm = np.degrees(self.fwhm[band_mask])
            offsets = np.degrees(self.offsets[band_mask])

            if units == "arcmin":
                fwhm *= 60
                offsets *= 60

            if units == "arcsec":
                fwhm *= 60
                offsets *= 3600

            band_color = HEX_CODE_LIST[iub]

            # nom_freq = self.dets.band_center[band_mask].mean()
            # band_res_arcmins = 2 * self.fwhm

            # 60 * np.degrees(
            #     1.22 * 2.998e8 / (1e9 * nom_freq * self.primary_size)
            # )

            # offsets_arcmins = 60 * np.degrees(self.offsets[band_mask])

            ax.add_collection(
                EllipseCollection(
                    widths=fwhm,
                    heights=fwhm,
                    angles=0,
                    units="xy",
                    facecolors=band_color,
                    edgecolors="k",
                    lw=1e-1,
                    alpha=0.5,
                    offsets=offsets,
                    transOffset=ax.transData,
                )
            )

            legend_handles.append(
                Patch(
                    label=f"{uband}, res = {fwhm.mean():.03f} {units}",
                    color=band_color,
                )
            )

            ax.scatter(*offsets.T, label=uband, s=5e-1, color=band_color)

        ax.set_xlabel(rf"$\theta_x$ offset ({units})")
        ax.set_ylabel(rf"$\theta_y$ offset ({units})")
        ax.legend(handles=legend_handles)
