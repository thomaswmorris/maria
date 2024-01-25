import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np
import pandas as pd

from .band import Band, BandList, generate_bands # noqa F401

here, this_filename = os.path.split(__file__)

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

REQUIRED_DET_CONFIG_KEYS = ["n", "band_center", "band_width"]

DET_COLUMN_TYPES = {
    "uid": "int",
    "band": "str",
    "nom_freq": "float",
    "offset_x": "float",
    "offset_y": "float",
    "baseline_x": "float",
    "baseline_y": "float",
    "baseline_z": "float",
    "pol_angle": "float",
    "efficiency": "float",
}


def generate_instrument_offsets(geometry, field_of_view, n):
    valid_instrument_types = ["flower", "hex", "square"]

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
        "Please specify a valid instrument type. Valid instrument types are:\n"
        + "\n".join(valid_instrument_types)
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



def generate_detectors(        
        bands_config: Mapping,
        field_of_view: float = 1,
        geometry: str = "hex",
        baseline: float = 0,
    ):

    dets = pd.DataFrame(columns=list(DET_COLUMN_TYPES.keys()), dtype=object)

    for band_key, band_config in bands_config.items():

        band_name = band_config.get("band_name", band_key)

        band_dets = pd.DataFrame(columns=dets.columns, index=np.arange(band_config["n_dets"]))
        band_dets.loc[:, 'band'] = band_name

        det_offsets_radians = np.radians(
            generate_instrument_offsets(geometry, field_of_view, len(band_dets))
        )

        # should we make another function for this?
        det_baselines_meters = generate_instrument_offsets(geometry, baseline, len(band_dets))

        # if randomize_offsets:
        #     np.random.shuffle(offsets_radians)  # this is a stupid function.

        band_dets.loc[:, "nom_freq"] = band_config.get("band_center")

        band_dets.loc[:, "offset_x"] = det_offsets_radians[:, 0]
        band_dets.loc[:, "offset_y"] = det_offsets_radians[:, 1]
        band_dets.loc[:, "baseline_x"] = det_baselines_meters[:, 0]
        band_dets.loc[:, "baseline_y"] = det_baselines_meters[:, 1]
        band_dets.loc[:, "baseline_z"] = 0 * det_baselines_meters[:, 1]

        band_dets.loc[:, "pol_angle"] = 0
        band_dets.loc[:, "efficiency"] = band_config.get("efficiency", 1.0)

        dets = pd.concat([dets, band_dets], axis=0)



    dets.loc[:, "uid"] = np.arange(len(dets))
    dets.index = np.arange(len(dets))

    for col, dtype in DET_COLUMN_TYPES.items():
        dets.loc[:, col] = dets.loc[:, col].astype(dtype)

    return dets

class Detectors:

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
        
    @classmethod
    def generate(
        cls,
        bands_config: Mapping,
        field_of_view: float = 1,
        geometry: str = "hex",
        baseline: float = 0,
    ):

        dets_df = generate_detectors(        
                bands_config=bands_config,
                field_of_view=field_of_view,
                geometry=geometry,
                baseline=baseline,
            )

        bands = generate_bands(bands_config=bands_config)


        return cls(df=dets_df, bands=bands)

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
    def ubands(self):
        return list(self.bands.keys())

    def passband(self, nu):
        _nu = np.atleast_1d(nu)

        PB = np.zeros((len(self.df), len(_nu)))

        for band in self.bands:
            PB[self.band == band.name] = band.passband(_nu)

        return PB
