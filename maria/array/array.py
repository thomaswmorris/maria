from __future__ import annotations

import copy
import glob
import logging
import os
import uuid
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from ..band import Band, BandList  # noqa
from ..beam import compute_angular_fwhm
from ..units import Quantity
from ..utils import HEX_CODE_LIST, compute_diameter, flatten_config, get_rotation_matrix_2d, read_yaml
from .generation import generate_2d_pattern

here, this_filename = os.path.split(__file__)

ALLOWED_ARRAY_KWARGS = [
    "band",
    "bands",
    "max_baseline",
    "baseline_offset",
    "baseline_spacing",
    "bath_temp",
    "beam_spacing",
    "field_of_view",
    "file",
    "focal_plane_offset",
    "key",
    "n",
    "n_col",
    "n_row",
    "name",
    "packing",
    "polarized",
    "primary_size",
    "shape",
]


DET_COLUMN_TYPES = {
    "array_name": str,
    "uid": str,
    "base_det_index": int,
    "band_name": str,
    "band_center": float,
    "sky_x": float,
    "sky_y": float,
    "baseline_x": float,
    "baseline_y": float,
    "baseline_z": float,
    "pol_angle": float,
    "pol_label": str,
    "primary_size": float,
    "bath_temp": float,
    "time_constant": float,
    "white_noise": float,
    "pink_noise": float,
    "efficiency": float,
}


ARRAY_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    key = os.path.split(path)[1].split(".")[0]
    ARRAY_CONFIGS[key] = read_yaml(path)
ARRAY_CONFIGS = flatten_config(ARRAY_CONFIGS)

all_arrays = list(ARRAY_CONFIGS.keys())


def get_array_config(key=None, **kwargs):
    c = {}
    if key:
        if key not in ARRAY_CONFIGS:
            raise KeyError(f"'{key}' is not a valid array name.")
        c = {"name": key, **ARRAY_CONFIGS[key]}
    c.update(kwargs)
    return c


def get_array(key):
    return Array.from_config(get_array_config(key=key))


class Array:
    def __init__(self, name: str, dets: pd.DataFrame, bands: BandList, config: dict = {}):
        self.name = name  # or str(uuid.uuid4())
        self.dets = dets
        self.dets.index = np.arange(len(self.dets.index))
        self.dets.loc[:, "array_name"] = self.name

        self.bands = BandList([band for band in bands if band.name in self.dets.band_name.values])
        self.config = config

        for band_attr, det_attr in {
            "center": "band_center",
            "width": "band_width",
        }.items():
            self.dets.loc[:, det_attr] = getattr(self, band_attr)

    def split(self):
        array_list = []
        for array_name in sorted(np.unique(self.dets.array_name.values)):
            array_dets = self.dets.loc[self.dets.array_name.values == array_name]
            array_bands = [band for band in self.bands if band.name in array_dets.band_name.values]
            array_list.append(Array(name=array_name, dets=array_dets, bands=array_bands))
        return ArrayList(array_list)

    def mask(self, **kwargs):
        mask = np.ones(len(self.dets)).astype(bool)
        for k, v in kwargs.items():
            mask &= self.dets.loc[:, k].values == v
        return mask

    def subset(self, **kwargs):
        return self._subset(self.mask(**kwargs))

    def _subset(self, mask):
        df = self.dets.loc[mask]
        return Array(name=self.name, dets=df, bands=[b for b in self.bands if b.name in df.band_name.values])  # noqa

    def one_detector_from_each_band(self):
        first_det_mask = np.isin(
            np.arange(self.n),
            np.unique(self.band_name, return_index=True)[1],
        )
        return self._subset(mask=first_det_mask)

    def outer(self):
        outer_dets_index = sp.spatial.ConvexHull(self.offsets).vertices
        outer_dets_mask = np.isin(np.arange(self.n), outer_dets_index)
        return self._subset(mask=outer_dets_mask)

    @property
    def n(self):
        return len(self.dets)

    @property
    def offsets(self):
        return np.c_[self.sky_x, self.sky_y]

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y, self.baseline_z]

    @property
    def field_of_view(self):
        return Quantity(compute_diameter(self.offsets), units="rad")

    @property
    def max_baseline(self):
        return Quantity(compute_diameter(self.baselines), units="m")

    @property
    def index(self):
        return self.dets.index.values

    @property
    def ubands(self):
        return list(self.bands.keys())

    @property
    def fwhm(self):
        """
        Returns the angular FWHM (in radians) at infinite distance.
        """
        return self.angular_fwhm(z=np.inf)

    def mueller(self):
        """
        For transforming Stoke's parameters. $theta$ = 0 is the horizontal
        """
        a = self.pol_angle
        m = np.stack(
            [
                np.where(np.isnan(a), np.sqrt(2), 1),
                np.where(np.isnan(a), 0, np.cos(2 * a)),
                np.where(np.isnan(a), 0, np.sin(2 * a)),
                np.zeros_like(a),
            ],
            axis=1,
        )
        return 0.5 * m[..., None] * m[..., None, :]

    def stokes_weight(self):
        return self.mueller()[:, 0]

    def angular_fwhm(self, z=np.inf):  # noqa F401
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
        PB = np.zeros((len(self.dets), len(_nu)))
        for band in self.bands:
            PB[self.band_name == band.name] = band.passband(_nu)
        return PB

    def __call__(self, band: str = None):
        mask = True
        if band:
            mask &= self.band_name == band
        return Array(name=self.name, dets=self.dets.loc[mask], bands=self.bands)

    def __getattr__(self, attr):
        if attr in self.dets.columns:
            return self.dets.loc[:, attr].values.astype(DET_COLUMN_TYPES[attr])

        if all(hasattr(band, attr) for band in self.bands):
            values = np.zeros(shape=self.n, dtype=float)
            for band in self.bands:
                values[self.band_name == band.name] = getattr(band, attr)
            return values

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __len__(self):
        return len(self.dets)

    @property
    def polarized(self):
        return not np.isnan(self.pol_angle).all()

    def filling(self):
        return {
            "n": self.n,
            "FOV": self.field_of_view,
            "baseline": self.max_baseline,
            "bands": f"[{','.join(self.bands.name)}]",
            "polarized": self.polarized,
        }

    def summary(self):
        series = pd.Series(self.filling())
        series.name = self.name
        return series

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.filling().items()])})"

    def __getitem__(self, key):
        return Array(name=self.name, dets=self.dets[key], bands=self.bands)

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls.from_config(kwargs)

    @classmethod
    def from_config(cls, c):
        config = copy.deepcopy(c)
        if "key" in config:
            config = get_array_config(key=config.pop("key"))
        config.update(c)

        logging.debug(f"Generating array with config {c}")

        bad_keys = [key for key in config if key not in ALLOWED_ARRAY_KWARGS]
        if bad_keys:
            raise ValueError(f"Bad kwargs {bad_keys}.")

        if "file" in config:
            df = pd.read_csv(f"{here}/{config['file']}")
        else:
            df = pd.DataFrame()

        if "primary_size" in df.columns:
            primary_sizes = list(np.unique(df.primary_size.values))
        elif "primary_size" in config:
            primary_sizes = list(np.atleast_1d(config["primary_size"]))
        else:
            raise ValueError("Missing array parameter 'primary_size'.")

        if "bands" in config:
            bands = BandList(config["bands"])
        elif "band" in config:
            bands = BandList([Band(config["band"])])
        else:
            raise ValueError("You must specify 'band' or 'bands' to generate an array.")

        max_resolution = max(
            [
                compute_angular_fwhm(primary_size, z=np.inf, nu=band.center)
                for band in bands
                for primary_size in primary_sizes
            ]
        )

        if "file" not in config:
            # we need:
            # - two of (n-like, field_of_view, beam_spacing), or
            # - two of (n-like, max_baseline, baseline_spacing)

            n_kwargs = {k: config.get(k) for k in ["n", "n_col", "n_row"] if config.get(k) is not None}
            n_explicit = ("n" in n_kwargs) or (("n_col" in n_kwargs) and ("n_row" in n_kwargs))
            n_focal_plane_kwargs = sum([n_explicit, "field_of_view" in config, "focal_plane_spacing" in config])
            n_baseline_kwargs = sum(
                [
                    n_explicit,
                    "max_baseline" in config,
                    "baseline_spacing" in config,
                ]
            )

            # this is not supposed to be elegant, it is supposed to be easy to understand
            if n_explicit:
                pattern_kwargs = {**n_kwargs}
                if ("field_of_view" in config) or ("beam_spacing" in config):
                    # we can only use one if n is explicit
                    mode = "focal_plane"
                    if "field_of_view" in config:
                        pattern_kwargs["max_diameter"] = np.radians(config["field_of_view"])
                    else:
                        pattern_kwargs["spacing"] = config.get("beam_spacing", 1.5) * max_resolution

                elif "max_baseline" in config:
                    mode = "baseline"
                    pattern_kwargs["max_diameter"] = config["max_baseline"]

                elif "baseline_spacing" in config:
                    mode = "baseline"
                    pattern_kwargs["spacing"] = config["baseline_spacing"]

                else:
                    raise ValueError(
                        "Invalid array spec: if detector counts are explicit, you must supply either 'field_of_view' or "
                        "either 'max_baseline' or 'baseline_spacing' for an interferometer array."
                    )

            else:
                pattern_kwargs = {}
                if "field_of_view" in config:
                    mode = "focal_plane"
                    pattern_kwargs["max_diameter"] = np.radians(config["field_of_view"])
                    pattern_kwargs["spacing"] = config.get("beam_spacing", 1.5) * max_resolution

                elif ("max_baseline" in config) and ("baseline_spacing" in config):
                    mode = "baseline"
                    pattern_kwargs["max_diameter"] = config["max_baseline"]
                    pattern_kwargs["spacing"] = config["baseline_spacing"]

                else:
                    raise ValueError(
                        "Invalid array spec: if detector counts are implicit, you must supply either 'field_of_view' or "
                        "both 'max_baseline' or 'baseline_spacing' for an interferometer array."
                    )

            X = generate_2d_pattern(
                **pattern_kwargs,
                shape=config.get("shape", "hexagon"),
                packing=config.get("packing", "triangular"),
            )

            if mode == "focal_plane":
                df = pd.DataFrame(X, columns=["sky_x", "sky_y"])
            else:
                df = pd.DataFrame(X, columns=["baseline_x", "baseline_y"])

            # if n_focal_plane_kwargs >= 2:
            #     X = generate_2d_pattern(
            #         **n_kwargs,
            #         shape=config.get("shape", "hexagon"),
            #         packing=config.get("packing", "triangular"),
            #         max_diameter=np.radians(config.get("field_of_view")),
            #         spacing=config.get("focal_plane_spacing"),
            #     )

            #     df = pd.DataFrame(X, columns=["sky_x", "sky_y"])

            # elif n_baseline_kwargs >= 2:
            #     X = generate_2d_pattern(
            #         **n_kwargs,
            #         shape=config.get("shape", "circle"),
            #         packing=config.get("packing", "sunflower"),
            #         max_diameter=config.get("max_baseline"),
            #         spacing=config.get("baseline_spacing"),
            #     )

            #     df = pd.DataFrame(X, columns=["baseline_x", "baseline_y"])

            # else:
            #     raise ValueError(
            #         "Invalid array spec: you must supply exactly two of [n, field_of_view, beam_spacing] for "
            #         "an array or exactly two of [n, max_baseline, baseline_spacing]."
            #     )

            df.loc[:, "bath_temp"] = config.get("bath_temp", 0)

        df.loc[:, "base_det_index"] = np.arange(len(df))

        for key in ["primary_size", "bath_temp"]:
            if (key in config) and (key not in df.columns):
                df.loc[:, key] = config[key]

        baseline_offset = config.get("baseline_offset", (0.0, 0.0, 0.0))
        focal_plane_offset = config.get("focal_plane_offset", (0.0, 0.0))

        for i in range(3):
            dim = "xyz"[i]
            if f"baseline_{dim}" not in df.columns:
                df.loc[:, f"baseline_{dim}"] = 0.0
            df.loc[:, f"baseline_{dim}"] += baseline_offset[i]
            if dim == "z":
                continue
            if f"sky_{dim}" not in df.columns:
                df.loc[:, f"sky_{dim}"] = 0.0
            df.loc[:, f"sky_{dim}"] += np.radians(focal_plane_offset[i])

        if config.get("polarized", False):
            df.loc[:, "pol_angle"] = np.random.uniform(
                low=0,
                high=np.pi,
                size=len(df),
            )
            df.loc[:, "pol_label"] = "A"

            other_df = df.copy()
            other_df.loc[:, "pol_angle"] = (df.pol_angle + np.pi / 2) % np.pi
            other_df.loc[:, "pol_label"] = "B"
            df = pd.concat([df, other_df])

        else:
            df.loc[:, "pol_angle"] = None

        if "band_name" not in df.columns:
            band_dfs = []
            for band in bands:
                band_df = df.copy()
                band_df.loc[:, "band_name"] = band.name
                band_dfs.append(band_df)
            df = pd.concat(band_dfs)

        df = df.sort_values(["band_name", "base_det_index"], ascending=True)

        for col in df.columns:
            df[col] = df[col].astype(DET_COLUMN_TYPES.get(col, str))

        return cls(dets=df, bands=bands, name=config.get("name", str(uuid.uuid4())[:8]))

    def plot(self, z=np.inf, plot_baseline="infer", plot_pol_angles=True):
        # if plot_baseline == "infer":
        #     plot_baseline = self.dets.max_baseline > 0

        # if plot_baseline:
        fig, (focal_ax, band_ax) = plt.subplots(1, 2, figsize=(10, 5), dpi=256, constrained_layout=True)

        i = 0
        legend_handles = []
        band_legend_handles = []

        focal_plane_units = Quantity(self.offsets, "rad").units
        resolution_units = Quantity(self.fwhm, "rad").units

        qnu_min, qnu_max = self.bands.nu_min, self.bands.nu_max

        qnu = Quantity(np.linspace(self.bands.nu_min, self.bands.nu_max, 1024), "Hz")
        qnu_min, qnu_max = qnu.value.min(), qnu.value.max()

        for ia, array in enumerate(self.split()):
            for ib, band in enumerate(array.bands):
                c = HEX_CODE_LIST[i % len(HEX_CODE_LIST)]

                band_array = array(band=band.name)

                fwhms = Quantity(band_array.angular_fwhm(z=z), "rad")
                offsets = Quantity(band_array.offsets, "rad")
                pol_angles = Quantity(band_array.pol_angle, "rad")
                # baselines = Quantity(band_array.baselines, "m")

                if plot_pol_angles and self.polarized:
                    dx = np.c_[-np.ones(band_array.n), np.ones(band_array.n)] / 2
                    dy = np.zeros((band_array.n, 2))

                    R = get_rotation_matrix_2d(pol_angles.rad)
                    dl = np.moveaxis(R @ np.stack([dx, dy], axis=1), 0, 2)
                    P = Quantity(offsets.rad.T[:, None] + fwhms.rad * dl, units="rad")
                    focal_ax.plot(*getattr(P, focal_plane_units), c=c, lw=2e0)

                focal_ax.add_collection(
                    EllipseCollection(
                        widths=getattr(fwhms, focal_plane_units),
                        heights=getattr(fwhms, focal_plane_units),
                        angles=0,
                        units="xy",
                        facecolors=c,
                        edgecolors="k",
                        lw=1e-1,
                        alpha=0.5,
                        offsets=getattr(offsets, focal_plane_units),
                        transOffset=focal_ax.transData,
                    ),
                )

                band_ax.plot(qnu.value, band.passband(qnu.Hz), color=c)

                legend_handles.append(
                    Patch(
                        label=f"{band.name} (n={band_array.n}, "
                        f"res={getattr(fwhms, resolution_units).mean():.01f} {resolution_units})",
                        color=c,
                    ),
                )
                band_legend_handles.append(Patch(label=rf"{band.name} ($\eta$={band.efficiency})", color=c))  # noqa

                focal_ax.scatter(
                    *getattr(offsets, focal_plane_units).T,
                    s=0,
                    color=c,
                )

                i += 1

        focal_ax.set_xlabel(rf"$\theta_x$ offset [{focal_plane_units}]")
        focal_ax.set_ylabel(rf"$\theta_y$ offset [{focal_plane_units}]")
        focal_ax.legend(handles=legend_handles, fontsize=8)

        band_ax.set_xlabel(rf"$\nu$ [${qnu.u['math_name']}$]")
        band_ax.set_ylabel(rf"$\tau(\nu)$")
        band_ax.legend(handles=band_legend_handles, fontsize=8)

        band_ax.plot([qnu_min, qnu_max], [0, 0], c="k", lw=0.5, ls=":")
        band_ax.set_xlim(qnu_min, qnu_max)

        xls, yls = focal_ax.get_xlim(), focal_ax.get_ylim()
        cen_x, cen_y = np.mean(xls), np.mean(yls)
        wid_x, wid_y = np.ptp(xls), np.ptp(yls)
        radius = 0.5 * np.maximum(wid_x, wid_y)

        margin = getattr(fwhms, focal_plane_units).max()

        focal_ax.set_xlim(cen_x - radius - margin, cen_x + radius + margin)
        focal_ax.set_ylim(cen_y - radius - margin, cen_y + radius + margin)


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
