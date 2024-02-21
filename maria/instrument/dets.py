import os
from collections.abc import Mapping

import matplotlib as mpl
import numpy as np
import pandas as pd

from .. import utils
from .bands import BandList, all_bands

here, this_filename = os.path.split(__file__)

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

REQUIRED_DET_CONFIG_KEYS = ["n", "band_center", "band_width"]

DET_COLUMN_TYPES = {
    "tag": "str",
    "uid": "str",
    "band": "str",
    "band_center": "float",
    "offset_x": "float",
    "offset_y": "float",
    "baseline_x": "float",
    "baseline_y": "float",
    "baseline_z": "float",
    "pol_angle": "float",
    "efficiency": "float",
    "primary_size": "float",
    "bath_temp": "float",
}

SUPPORTED_GEOMETRIES = ["flower", "hex", "square", "circle"]


def hex_packed_circle_offsets(n):
    """
    Returns an array of $n$ hexagonal offsets from the origin with diameter 1.
    """

    h = int(np.ceil((np.sqrt(12 * n - 3) + 3) / 6))

    side = np.arange(-h, h + 1, dtype=float)
    x, y = np.meshgrid(side, side)

    x[1::2] -= 0.5
    y *= np.sqrt(3) / 2

    offsets = np.c_[x.ravel(), y.ravel()]

    distance_squared = np.sum(offsets**2, axis=1)

    subset_index = np.argsort(distance_squared)[:n]

    return 0.5 * offsets[subset_index] / np.sqrt(distance_squared[subset_index[-1]])


def generate_offsets(n, shape="hex"):
    if shape not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"'shape' must be one of {SUPPORTED_GEOMETRIES}.")

    if shape == "circle":
        return hex_packed_circle_offsets(n).T

    if shape == "hex":
        angles = np.linspace(0, 2 * np.pi, 6 + 1)[1:] + np.pi / 2
        z = np.array([0])
        layer = 0
        while len(z) < n:
            for angle in angles:
                for _z in layer * np.exp(1j * angle) + np.arange(layer) * np.exp(
                    1j * (angle + 2 * np.pi / 3)
                ):
                    z = np.r_[z, _z]
            layer += 1
        z -= z.mean()
        z *= 0.5 / np.abs(z).max()

        return np.c_[np.real(np.array(z[:n])), np.imag(np.array(z[:n]))].T

    if shape == "flower":
        golden_ratio = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        z = np.zeros(n).astype(complex)
        for i in range(n):
            z[i] = np.sqrt((i / (n - 1)) * 2) * np.exp(1j * golden_ratio * i)
        z *= 0.5 / np.abs(z.max())
        return np.c_[np.real(z), np.imag(z)].T

    if shape == "square":
        side = np.linspace(-0.5, 0.5, int(np.ceil(np.sqrt(n))))
        DX, DY = np.meshgrid(side, side)
        return np.c_[DX.ravel()[:n], DY.ravel()[:n]].T


def generate_dets(
    n: int,
    bands: list,
    field_of_view: float = 0.0,
    array_offset: tuple = (0.0, 0.0),
    array_shape: tuple = "hex",
    baseline_offset: tuple = (0.0, 0.0, 0.0),
    baseline_diameter: float = 0.0,
    baseline_shape: str = "flower",
    **kwargs,
):
    dets = pd.DataFrame()

    detector_offsets = field_of_view * generate_offsets(n=n, shape=array_shape)

    baselines = baseline_diameter * generate_offsets(n=n, shape=baseline_shape)

    for band in bands:
        band_dets = pd.DataFrame(
            index=np.arange(n), columns=["band", "offset_x", "offset_y"], dtype=float
        )

        band_dets.loc[:, "band"] = band
        band_dets.loc[:, "offset_x"] = array_offset[0] + detector_offsets[0]
        band_dets.loc[:, "offset_y"] = array_offset[1] + detector_offsets[1]

        band_dets.loc[:, "baseline_x"] = baseline_offset[0] + baselines[0]
        band_dets.loc[:, "baseline_y"] = baseline_offset[1] + baselines[1]
        band_dets.loc[:, "baseline_z"] = baseline_offset[2]

        dets = pd.concat([dets, band_dets])

    return dets


def validate_band_config(band):
    if any([key not in band for key in ["center"]]):
        raise ValueError("The band center must be specified")


def parse_bands_config(bands):
    """
    There are many ways to specify bands, and this handles them.
    """
    parsed_band_config = {}

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

    if isinstance(bands, Mapping):
        for name, band in bands.items():
            validate_band_config(band)
            parsed_band_config[name] = band

    return parsed_band_config


class Detectors:
    @classmethod
    def from_config(cls, config):
        """
        Instantiate detectors from a config. We pass the whole config and not just config["dets"] so
        that the detectors can inherit instrument parameters if need be.
        """

        config["dets"] = utils.io.flatten_config(config["dets"])

        bands_config = {}

        df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in DET_COLUMN_TYPES.items()}
        )

        for tag in config["dets"]:
            tag_dets_config = config["dets"][tag]

            if "file" in tag_dets_config:
                # if a file is supplied, assume it's a csv and read it in
                tag_df = pd.read_csv(f'{here}/{tag_dets_config["file"]}', index_col=0)

                # if no bands were supplied, get them from the table and hope they're registered
                if "bands" not in tag_dets_config:
                    tag_dets_config["bands"] = sorted(np.unique(tag_df.band.values))

                tag_dets_config["bands"] = parse_bands_config(tag_dets_config["bands"])

            else:
                tag_dets_config["bands"] = parse_bands_config(tag_dets_config["bands"])
                tag_df = generate_dets(**tag_dets_config)

            fill_level = int(np.log(len(tag_df) - 1) / np.log(10) + 1)

            uid_predix = f"{tag}-" if tag else ""
            uids = [
                f"{uid_predix}{str(i).zfill(fill_level)}" for i in range(len(tag_df))
            ]

            tag_df.insert(0, "uid", uids)
            tag_df.insert(1, "tag", tag)

            df = pd.concat([df, tag_df])

            for band, band_config in tag_dets_config["bands"].items():
                bands_config[band] = band_config

        df.index = np.arange(len(df))

        for col in ["primary_size"]:
            if df.loc[:, col].isna().any():
                df.loc[:, col] = config[col]

        for col in [
            "offset_x",
            "offset_y",
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
            df.loc[df.band == band.name, "band_center"] = band.center
            df.loc[df.band == band.name, "efficiency"] = band.efficiency

        return cls(df=df, bands=bands)

    def __repr__(self):
        return self.df.__repr__()

    def __repr_html__(self):
        return self.df.__repr_html__()

    def __init__(self, df: pd.DataFrame, bands: dict = {}):
        self.df = df
        self.bands = bands

    def __getattr__(self, attr):
        if attr in self.df.columns:
            return self.df.loc[:, attr].values.astype(DET_COLUMN_TYPES[attr])

    # def __call__(self, band=None):
    #     if band is not None:
    #         return self.subset(band=band)

    def subset(self, band=None):
        bands = BandList(self.bands[band])
        mask = self.band == band
        return Detectors(bands=bands, df=self.df.loc[mask])

    @property
    def n(self):
        return len(self.df)

    @property
    def offset(self):
        return np.c_[self.offset_x, self.offset_y]

    # @property
    # def band_center(self):
    #     centers = np.zeros(shape=self.n)
    #     for band in self.bands:
    #         centers[self.band == band.name] = band.center

    #     return centers

    # @property
    # def band_width(self):
    #     widths = np.zeros(shape=self.n)
    #     for band in self.bands:
    #         widths[self.band == band.name] = band.width

    #     return widths

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
            PB[self.band == band.name] = band.passband(_nu)

        return PB
