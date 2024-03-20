import os
import warnings
from collections.abc import Mapping

import matplotlib as mpl
import numpy as np
import pandas as pd

from .. import utils
from ..constants import c
from .bands import BandList, all_bands

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
    "abs_cal_rj": "float",
    "time_constant": "float",
    "white_noise": "float",
    "pink_noise": "float",
    "efficiency": "float",
    "abs_cal_rj": "float",
}

SUPPORTED_ARRAY_PACKINGS = ["hex", "square", "sunflower"]
SUPPORTED_ARRAY_SHAPES = ["hex", "square", "circle"]


def generate_array_offsets(
    n: int = None,
    diameter: float = None,
    spacing: float = None,
    packing: str = "hex",
    shape: str = "hex",
) -> np.array:
    n_params = (n is not None) + (diameter is not None) + (spacing is not None)
    if n_params < 2:
        raise ValueError("You must specify at least two of (n, diameter, spacing)")

    if n_params == 3:
        warnings.warn(
            "'n', 'diameter', and 'spacing' were all supplied for array generation. Ignoring spacing parameter."
        )

    if packing not in SUPPORTED_ARRAY_PACKINGS:
        raise ValueError(f"'packing' must be one of {SUPPORTED_ARRAY_PACKINGS}")

    if shape not in SUPPORTED_ARRAY_SHAPES:
        raise ValueError(f"'shape' must be one of {SUPPORTED_ARRAY_SHAPES}")

    if n is None:
        r = int(np.ceil(diameter / spacing / 2))

        if shape == "hex":
            # a packed hexagon of radius $r$ has $n = 3r^2 - 3r + 1$ points in it
            n = int(np.ceil(3 * r**2 - 3 * r + 1))

        if shape == "circle":
            n = int(np.ceil(np.pi * r**2))

        if shape == "square":
            n = 4 * r**2

    if spacing is None:
        if shape == "hex":
            # a packed hexagon of radius $r$ has $n = 3r^2 - 3r + 1$ points in it
            r = int((np.sqrt(12 * n - 3) + 3) / 6)
            spacing = diameter / (2 * r)

        if shape == "circle":
            spacing = np.pi * diameter / (4 * np.sqrt(n))

        if shape == "square":
            spacing = diameter / np.sqrt(n)

    super_n = n * 2

    if packing == "square":
        s = int(np.ceil(np.sqrt(super_n)))
        side = np.arange(-s, s + 1, dtype=float)
        x, y = [foo.ravel() for foo in np.meshgrid(side, side)]

    if packing == "hex":
        s = int(np.ceil((np.sqrt(12 * super_n - 3) + 3) / 6))
        side = np.arange(-s, s + 1, dtype=float)
        x, y = np.meshgrid(side, side)
        y[:, ::2] -= 0.5
        x *= np.sqrt(3) / 2
        x, y = x.ravel(), y.ravel()

    if packing == "sunflower":
        i = np.arange(super_n)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        x = 0.586 * np.sqrt(i) * np.cos(golden_angle * i)
        y = 0.586 * np.sqrt(i) * np.sin(golden_angle * i)

    n_sides = {"square": 4, "hex": 6, "circle": 256}[shape]

    r = np.sqrt(x**2 + y**2)
    p = np.arctan2(y, x)
    ngon_distance = r * np.cos(np.arcsin(np.sin(n_sides / 2 * p)) * 2 / n_sides)

    subset_index = np.argsort(ngon_distance)[:n]

    return spacing * np.c_[x[subset_index], y[subset_index]]


def generate_dets(
    bands: list,
    n: int = None,
    field_of_view: float = None,
    array_spacing: float = None,
    array_packing: tuple = "hex",
    array_shape: tuple = "circle",
    array_offset: tuple = (0.0, 0.0),
    baseline_diameter: float = 0,
    baseline_packing: str = "sunflower",
    baseline_shape: str = "circle",
    baseline_offset: tuple = (0.0, 0.0, 0.0),
    polarized: bool = False,
    bath_temp: float = 0,
):
    dets = pd.DataFrame()

    detector_offsets = generate_array_offsets(
        n=n,
        diameter=field_of_view,
        spacing=array_spacing,
        packing=array_packing,
        shape=array_shape,
    )

    baselines = generate_array_offsets(
        n=len(detector_offsets),
        diameter=baseline_diameter,
        packing=baseline_packing,
        shape=baseline_shape,
    )

    if polarized:
        pol_angles = np.random.uniform(low=0, high=360, size=len(detector_offsets))
        pol_labels = np.r_[["A" for _ in pol_angles], ["B" for _ in pol_angles]]
        pol_angles = np.r_[pol_angles, (pol_angles + 90) % 360]
        detector_offsets = np.r_[detector_offsets, detector_offsets]
        baselines = np.r_[baselines, baselines]

    else:
        pol_angles = np.zeros(len(detector_offsets))
        pol_labels = ["A" for i in pol_angles]

    for band in bands:
        band_dets = pd.DataFrame(
            index=np.arange(len(detector_offsets)),
            columns=["band_name", "sky_x", "sky_y"],
            dtype=float,
        )

        band_dets.loc[:, "band_name"] = band
        band_dets.loc[:, "sky_x"] = array_offset[0] + detector_offsets[:, 0]
        band_dets.loc[:, "sky_y"] = array_offset[1] + detector_offsets[:, 1]

        band_dets.loc[:, "baseline_x"] = baseline_offset[0] + baselines[:, 0]
        band_dets.loc[:, "baseline_y"] = baseline_offset[1] + baselines[:, 1]
        band_dets.loc[:, "baseline_z"] = baseline_offset[2]

        band_dets.loc[:, "bath_temp"] = bath_temp
        band_dets.loc[:, "pol_angle"] = pol_angles
        band_dets.loc[:, "pol_label"] = pol_labels

        dets = pd.concat([dets, band_dets])

    return dets


def validate_band_config(band):
    if any([key not in band for key in ["center", "width"]]):
        raise ValueError("The band's center and width must be specified!")


def parse_bands_config(bands):
    """
    There are many ways to specify bands, and this handles them.
    """
    parsed_band_config = {}

    if isinstance(bands, Mapping):
        for name, band in bands.items():
            validate_band_config(band)
            parsed_band_config[name] = band
        return parsed_band_config

    if isinstance(bands, list):
        for band in bands:
            if isinstance(band, str):
                if band not in all_bands:
                    raise ValueError(f'Band "{band}" is not supported.')
                parsed_band_config[band] = all_bands[band]

            if isinstance(band, Mapping):
                validate_band_config(band)
                name = band.get("name", f'f{int(band["center"]):>03}')
                parsed_band_config[name] = band

    return parsed_band_config


class Detectors:
    @classmethod
    def from_config(cls, config):
        """
        Instantiate detectors from a config. We pass the whole config and not just config["dets"] so
        that the detectors can inherit instrument parameters if need be (e.g. the size of the primary aperture).
        """

        config["dets"] = utils.io.flatten_config(config["dets"])

        bands_config = {}

        df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in DET_COLUMN_TYPES.items()}
        )

        for array in config["dets"]:
            array_dets_config = config["dets"][array]

            if "band" in array_dets_config:
                array_dets_config["bands"] = [array_dets_config.pop("band")]

            if "file" in array_dets_config:
                # if a file is supplied, assume it's a csv and read it in
                array_df = pd.read_csv(
                    f'{here}/{array_dets_config["file"]}', index_col=0
                )

                # if no bands were supplied, get them from the table and hope they're registered
                if "bands" not in array_dets_config:
                    array_dets_config["bands"] = sorted(np.unique(array_df.band.values))

                array_dets_config["bands"] = parse_bands_config(
                    array_dets_config["bands"]
                )

            else:
                array_dets_config["bands"] = parse_bands_config(
                    array_dets_config["bands"]
                )

                if "beam_spacing" in array_dets_config:
                    if len(array_dets_config["bands"]) > 1:
                        raise ValueError(
                            "'beam_spacing' parameter is unhandled for an detector array with multiple bands."
                        )

                    nu = (
                        1e9
                        * array_dets_config["bands"][
                            list(array_dets_config["bands"].keys())[0]
                        ]["center"]
                    )
                    beam_resolution_degrees = np.degrees(
                        1.22 * (c / nu) / config["primary_size"]
                    )
                    array_dets_config[
                        "array_spacing"
                    ] = beam_resolution_degrees * array_dets_config.pop("beam_spacing")

                array_df = generate_dets(**array_dets_config)

            # add leading zeros to detector uids
            fill_level = int(np.log(np.maximum(len(array_df) - 1, 1)) / np.log(10) + 1)

            uid_predix = f"{array}-" if array else ""
            uids = [
                f"{uid_predix}{str(i).zfill(fill_level)}" for i in range(len(array_df))
            ]

            array_df.insert(0, "uid", uids)
            array_df.insert(1, "array", array)

            df = pd.concat([df, array_df])

            for band, band_config in array_dets_config["bands"].items():
                bands_config[band] = band_config

        df.index = np.arange(len(df))

        for col in ["primary_size"]:
            if df.loc[:, col].isna().any():
                df.loc[:, col] = config[col]

        for col in [
            "sky_x",
            "sky_y",
            "baseline_x",
            "baseline_y",
            "baseline_z",
            "pol_angle",
            "bath_temp",
        ]:
            if df.loc[:, col].isna().any():
                df.loc[:, col] = 0

        bands = BandList.from_config(bands_config)

        for band in bands:
            df.loc[df.band_name == band.name, "band_center"] = band.center
            df.loc[df.band_name == band.name, "efficiency"] = band.efficiency

        return cls(df=df, bands=bands)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def __init__(self, df: pd.DataFrame, bands: dict = {}):
        self.df = df
        self.bands = bands

        for attr in [
            "time_constant",
            "white_noise",
            "pink_noise",
            "efficiency",
            "abs_cal_rj",
        ]:
            values = np.zeros(shape=self.n)
            for band in self.bands:
                values[self.band_name == band.name] = getattr(band, attr)
            self.df.loc[:, attr] = values

    def __getattr__(self, attr):
        if attr in self.df.columns:
            return self.df.loc[:, attr].values.astype(DET_COLUMN_TYPES[attr])

        raise AttributeError(f"'Detectors' object has no attribute '{attr}'")

    def subset(self, band_name=None):
        bands = BandList([self.bands[band_name]])
        return Detectors(bands=bands, df=self.df.loc[self.band_name == band_name])

    @property
    def n(self):
        return len(self.df)

    @property
    def offset(self):
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
