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
}

SUPPORTED_GEOMETRIES = ["flower", "hex", "square"]


def generate_offsets(n, geometry="hex"):
    if geometry not in SUPPORTED_GEOMETRIES:
        raise ValueError(f"'geometry' must be one of {SUPPORTED_GEOMETRIES}.")

    if geometry == "hex":
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

    if geometry == "flower":
        golden_ratio = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        z = np.zeros(n).astype(complex)
        for i in range(n):
            z[i] = np.sqrt((i / (n - 1)) * 2) * np.exp(1j * golden_ratio * i)
        z *= 0.5 / np.abs(z.max())
        return np.c_[np.real(z), np.imag(z)].T

    if geometry == "square":
        side = np.linspace(-0.5, 0.5, int(np.ceil(np.sqrt(n))))
        DX, DY = np.meshgrid(side, side)
        return np.c_[DX.ravel()[:n], DY.ravel()[:n]].T


def generate_dets(
    n: int = 1,
    field_of_view: float = 0.0,
    boresight_offset: tuple = (0.0, 0.0),
    detector_geometry: tuple = "hex",
    baseline_offset: tuple = (0.0, 0.0, 0.0),
    baseline_diameter: float = 0.0,
    baseline_geometry: str = "flower",
    bands: list = [],
):
    dets = pd.DataFrame()

    detector_offsets = field_of_view * generate_offsets(n=n, geometry=detector_geometry)

    baselines = baseline_diameter * generate_offsets(n=n, geometry=baseline_geometry)

    for band in bands:
        band_dets = pd.DataFrame(
            index=np.arange(n), columns=["band", "offset_x", "offset_y"], dtype=float
        )

        band_dets.loc[:, "band"] = band
        band_dets.loc[:, "offset_x"] = boresight_offset[0] + detector_offsets[0]
        band_dets.loc[:, "offset_y"] = boresight_offset[1] + detector_offsets[1]

        band_dets.loc[:, "baseline_x"] = baseline_offset[0] + baselines[0]
        band_dets.loc[:, "baseline_y"] = baseline_offset[1] + baselines[1]
        band_dets.loc[:, "baseline_z"] = baseline_offset[2]

        dets = pd.concat([dets, band_dets])

    return dets


class Detectors:
    @classmethod
    def from_config(cls, config):
        dets_config = utils.io.flatten_config(config["dets"])

        df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in DET_COLUMN_TYPES.items()}
        )

        for tag, dets_config in utils.io.flatten_config(config["dets"]).items():
            if "file" in dets_config:
                tag_df = pd.read_csv(f'{here}/{dets_config["file"]}', index_col=0)

            else:
                tag_df = generate_dets(**dets_config)

            fill_level = int(np.log(len(tag_df) - 1) / np.log(10) + 1)
            tag_df.insert(
                0,
                "uid",
                [f"{tag}_{str(i).zfill(fill_level)}" for i in range(len(tag_df))],
            )
            tag_df.insert(1, "tag", tag)

            df = pd.concat([df, tag_df])

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
        ]:
            if df.loc[:, col].isna().any():
                df.loc[:, col] = 0

        # get the bands
        bands_config = {}
        for band in sorted(np.unique(df.band)):
            if isinstance(band, Mapping):
                bands_config[band] = band
            if isinstance(band, str):
                bands_config[band] = all_bands[band]

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

    def __call__(self, band=None):
        if band is not None:
            return self.subset(band=band)

    def subset(self, band=None):
        bands = BandList(self.bands[band])
        mask = self.band == band
        return Detectors(bands=bands, df=self.df.loc[mask])

    @property
    def n(self):
        return len(self.df)

    @property
    def band_center(self):
        centers = np.zeros(shape=self.n)
        for band in self.bands:
            centers[self.band == band.name] = band.center

        return centers

    @property
    def band_width(self):
        widths = np.zeros(shape=self.n)
        for band in self.bands:
            widths[self.band == band.name] = band.width

        return widths

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
