import glob
import logging
import os
import time as ttime

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from ...io import flatten_config, read_yaml
from ...units import Angle
from ...utils import compute_diameter
from ..band import BandList
from ..beam import compute_angular_fwhm
from ..detectors import Detectors

# from ..detectors import Detectors


here, this_filename = os.path.split(__file__)

SUPPORTED_ARRAY_PACKINGS = ["hex", "square", "sunflower"]
SUPPORTED_ARRAY_SHAPES = ["hex", "square", "circle"]

logger = logging.getLogger("maria")

ARRAY_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    key = os.path.split(path)[1].split(".")[0]
    ARRAY_CONFIGS[key] = read_yaml(path)
ARRAY_CONFIGS = flatten_config(ARRAY_CONFIGS)

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


def generate_2d_offsets(n, packing="hex", shape="circle", normalize=False):
    """
    Generate a scatter of $n$ points with some pattern.
    These points are spread such that each is a unit of distance away from its nearest neighbor.
    """

    n = int(np.maximum(n, 1))
    bigger_n = 2 * n

    if packing == "square":
        s = int(np.ceil(np.sqrt(bigger_n)))
        side = np.arange(-s, s + 1, dtype=float)
        x, y = [foo.ravel() for foo in np.meshgrid(side, side)]

    elif packing == "hex":
        s = int(np.ceil((np.sqrt(12 * bigger_n - 3) + 3) / 6))
        side = np.arange(-s, s + 1, dtype=float)
        x, y = np.meshgrid(side, side)
        y[:, ::2] -= 0.5
        x *= np.sqrt(3) / 2
        x, y = x.ravel(), y.ravel()

    elif packing == "sunflower":
        i = np.arange(bigger_n)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        x = 0.5966 * np.sqrt(i) * np.cos(golden_angle * i)
        y = 0.5966 * np.sqrt(i) * np.sin(golden_angle * i)

    else:
        raise ValueError(
            "Supported offset packings are ['square', 'hex', or 'sunflower']."
        )

    n_sides = {"square": 4, "hex": 6, "circle": 256}[shape]

    r = np.sqrt(x**2 + y**2)
    p = np.arctan2(y, x)
    ngon_distance = r * np.cos(np.arcsin(np.sin(n_sides / 2 * p)) * 2 / n_sides)

    subset_index = np.argsort(ngon_distance)[:n]

    offsets = np.c_[x[subset_index], y[subset_index]]

    if normalize:
        hull_pts = (
            offsets[ConvexHull(offsets).vertices] if len(offsets) > 16 else offsets
        )
        span = cdist(hull_pts, hull_pts, metric="euclidean").max()
        offsets /= span if span > 0 else 1.0

    return offsets


def generate_2d_offsets_from_diameter(
    diameter, packing="hex", shape="circle", tol=1e-2, max_iterations=32
):
    n = np.square(diameter)
    span = 0

    for _ in range(max_iterations):
        offsets = generate_2d_offsets(n=n, packing=packing, shape=shape)
        hull_pts = (
            offsets[ConvexHull(offsets).vertices] if len(offsets) > 16 else offsets
        )
        span = cdist(hull_pts, hull_pts, metric="euclidean").max()

        n *= np.square(diameter / span)

        if np.abs(span - diameter) / diameter < tol:
            return offsets

    return offsets


def get_array(key):
    return Array.get(key)


def get_array_config(key):
    if key not in ARRAY_CONFIGS:
        raise KeyError(f"'{key}' is not a valid array name.")
    return ARRAY_CONFIGS[key]


class Array:
    @classmethod
    def generate(
        cls,
        name: str,
        bands: list,
        df: pd.DataFrame = None,
        n: int = None,
        primary_size: float = 10.0,
        field_of_view: float = 0.0,
        beam_spacing: float = 1.2,
        array_packing: tuple = "hex",
        array_shape: tuple = "circle",
        array_offset: tuple = (0.0, 0.0),
        baseline_diameter: float = 0,
        baseline_packing: str = "sunflower",
        baseline_shape: str = "circle",
        baseline_offset: tuple = (0.0, 0.0, 0.0),
        polarization: str = None,
        bath_temp: float = 0,
    ):
        """
        Generate the things that aren't supplied.
        """

        if df is None:
            df = pd.DataFrame()

        if len(df) > 0:
            n = len(df)

        start_time = ttime.monotonic()
        bands = BandList(bands)

        band_centers = [band.center for band in bands]
        resolutions = [
            compute_angular_fwhm(primary_size, z=np.inf, nu=band.center)
            for band in bands
        ]
        detector_spacing = beam_spacing * np.max(resolutions)

        if ("offset_x" in df.columns) and ("offset_y" in df.columns):
            ...

        else:
            if n is None:
                topological_diameter = np.radians(field_of_view) / detector_spacing
                if topological_diameter > 1:
                    offsets = detector_spacing * generate_2d_offsets_from_diameter(
                        diameter=topological_diameter,
                        packing=array_packing,
                        shape=array_shape,
                    )
                else:
                    n = 1  # what?

            if n is not None:
                if field_of_view is not None:
                    offsets = np.radians(field_of_view) * generate_2d_offsets(
                        n=n, packing=array_packing, shape=array_shape, normalize=True
                    )
                else:
                    if len(resolutions) > 1:
                        logger.warning(
                            "Subarray has more than one band. "
                            f"Generating detector spacing based on the lowest frequency ({np.min(band_centers):.01f}) GHz."
                        )
                    offsets = detector_spacing * generate_2d_offsets(
                        n=n, packing=array_packing, shape=array_shape
                    )
            df.loc[:, "sky_x"] = np.radians(array_offset[0]) + offsets[:, 0]
            df.loc[:, "sky_y"] = np.radians(array_offset[1]) + offsets[:, 1]

        if (
            ("baseline_x" in df.columns)
            and ("baseline_y" in df.columns)
            and ("baseline_z" in df.columns)
        ):
            ...

        else:
            baselines = baseline_diameter * generate_2d_offsets(
                n=len(offsets),
                packing=baseline_packing,
                shape=baseline_shape,
                normalize=True,
            )

            df.loc[:, "baseline_x"] = baseline_offset[0] + baselines[:, 0]
            df.loc[:, "baseline_y"] = baseline_offset[1] + baselines[:, 1]
            df.loc[:, "baseline_z"] = baseline_offset[2]

        if "primary_size" not in df.columns:
            df.loc[:, "primary_size"] = primary_size

        if "pol_angle" in df.columns:
            ...

        else:
            df.loc[:, "pol_angle"] = np.random.uniform(
                low=0, high=2 * np.pi, size=len(offsets)
            )
            df.loc[:, "pol_label"] = "A"

            if polarization:
                other_df = df.copy()
                other_df.loc[:, "pol_angle"] = (df.pol_angle + np.pi / 2) % (2 * np.pi)
                other_df.loc[:, "pol_label"] = "B"
                df = pd.concat([df, other_df])

        # offsets += np.radians(1e-1 / 3600) * np.random.standard_normal()
        # baselines += np.radians(1e-1 / 3600) * np.random.standard_normal()

        if "band_name" not in df.columns:
            band_dfs = []

            for band in bands:
                band_df = df.copy()
                band_df.loc[:, "band_name"] = band.name
                band_dfs.append(band_df)

            df = pd.concat(band_dfs)

        df.loc[:, "bath_temp"] = bath_temp

        # logger.debug(f"generated detectors in {ttime.monotonic() - start_time:.02e} s")

        dets = Detectors(df=df, bands=bands)

        logger.debug(
            f"generated array '{name}' with {len(dets)} detectors in {int(1e3 * (ttime.monotonic() - start_time))} ms"
        )

        return cls(name=name, dets=dets)

    @classmethod
    def get(cls, key):
        return cls.generate(**get_array_config(key))

    @classmethod
    def from_config(cls, key=None, **kwargs):
        config = ARRAY_CONFIGS[key].copy() if key else {}
        config.update(kwargs)

        logger.debug(f"generating array with config {config}")

        if "file" in config:
            path = config.pop("file")
            config["df"] = pd.read_csv(f"{here}/{path}")
            # config["df"] = pd.read_csv(path)

            return cls.generate(**config)
        else:
            return cls.generate(**config)

        return cls()

    def __init__(self, name, dets):
        self.name = name

        # if isinstance(bands, BandList):
        #     self.bands = bands
        # elif isinstance(bands, list):
        #     self.bands = BandList(bands)
        # else:
        #     raise TypeError("bands")

        self.dets = dets

        self.dets.df.loc[:, "array_name"] = name

        for band in self.dets.bands:
            band_mask = self.dets.band_name == band.name
            for attr in ["center"]:
                self.dets.df.loc[band_mask, f"band_{attr}"] = getattr(band, attr)

    # def __getattr__(self, attr):
    #     if attr in self.dets.df.columns:
    #         return self.dets.df.loc[:, attr].values.astype(DET_COLUMN_TYPES[attr])

    #     if all(hasattr(band, attr) for band in self.bands):
    #         values = np.zeros(shape=self.n, dtype=float)
    #         for band in self.bands:
    #             values[self.band_name == band.name] = getattr(band, attr)
    #         return values

    #     raise AttributeError(f"'Detectors' object has no attribute '{attr}'")

    # def subset(self, band_name=None):
    #     bands = BandList([self.bands[band_name]])
    #     return Detectors(bands=bands, df=self.dets.loc[self.band_name == band_name])

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
        return Angle(compute_diameter(self.offsets))

    @property
    def max_baseline(self):
        return compute_diameter(self.baselines)

    @property
    def __len__(self):
        return len(self.dets)

    @property
    def index(self):
        return self.dets.index.values

    # def passband(self, nu):
    #     _nu = np.atleast_1d(nu)

    #     PB = np.zeros((len(self.dets), len(_nu)))

    #     for band in self.bands:
    #         PB[self.band_name == band.name] = band.passband(_nu)

    #     return PB

    # def __repr__(self):
    #     fov = self.field_of_view
    #     return f"Array(name={self.name}, n={len(self.dets)}, fov={getattr(fov, fov.units):.03f}
    #  {fov.units}, baseline={self.max_baseline:.01f} m)"


class ArrayList:
    def __init__(self, arrays: list):
        if not isinstance(arrays, list):
            raise TypeError("'arrays' must be a list.")

        self.arrays = arrays

        self.combine()

    def combine(self):
        self.dets = Detectors(
            df=pd.concat([a.dets.df for a in self.arrays]), bands=self.bands
        )

    @property
    def bands(self):
        bands = []
        for array in self.arrays:
            for band in array.dets.bands:
                if band not in bands:
                    bands.append(band)
        return BandList(bands)

    @property
    def summary(self):
        df = pd.DataFrame(columns=["n", "center", "baseline", "bands"])

        for array in self.arrays:
            df.loc[array.name, "n"] = array.dets.n
            df.loc[array.name, "center"] = tuple(
                np.degrees(array.dets.offsets.mean(axis=0)).round(2)
            )
            df.loc[array.name, "baseline"] = tuple(
                array.dets.baselines.mean(axis=0).round(2)
            )
            df.loc[array.name, "bands"] = ", ".join(
                list(array.dets.bands.summary.loc[:, "name"])
            )

        return df

    def __repr__(self):
        return self.summary.__repr__()

    def _repr_html_(self):
        return self.summary._repr_html_()

    def __iter__(self):  # it has to be called this
        return iter(self.arrays)  # return the list's iterator

    def __len__(self):
        return len(self.arrays)
