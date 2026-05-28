from __future__ import annotations

import logging
import os
import re

import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import ListedColormap

from ..coords import frames
from ..io import fetch, read_fits_map
from ..units import Quantity, parse_units
from .base import VALID_MAP_QUANTITIES, Map, concatenate  # noqa
from .healpix import HEALPixMap  # noqa
from .projection import ProjectionMap  # noqa
from .transfer import TransferFunction, compute_transfer_function, plot_transfer_function  # noqa

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

with open(f"{here}/maps.txt", "r") as f:
    all_maps = f.read().splitlines()

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/../plotting/Planck_Parchment_RGB.txt") / 255.0,
    name="cmb",
)
cmb_cmap.set_bad("white")
mpl.colormaps.register(cmb_cmap)

MAP_SIZE_KWARGS = ["xi", "eta", "width", "height", "xi_res", "eta_res", "resolution"]
VALID_MAP_KWARGS = ["stokes", "nu", "t", "center", "frame", "units", "beam", *MAP_SIZE_KWARGS]

FITS_KEYWORD_ALIASES = {
    "frame": ["FRAME"],
    "units": ["BUNIT", "BUNITS"],
    "nu": ["FREQ", "RESTFREQ"],
}

AXIS_MAPPING = {
    "nu": {
        "aliases": ["FREQ", "NU"],
        "default_units": "Hz",
    },
    "v": {
        "aliases": ["VRAD", "VELO"],
        "default_units": "m/s",
    },
}


def get(name: str, **map_kwargs):
    return load(fetch(name), **map_kwargs)


def load(filename: str, index: int = None, format: str = "auto", **map_kwargs) -> Map:
    if format == "auto":
        format = filename.split(".")[-1].lower()

    if format == "fits":
        data, axis_mask, kwargs, header = read_fits_map(filename, index=index, strict=False)
        data = np.squeeze(data, tuple([i for i in range(data.ndim) if not axis_mask[i]]))
    elif format == "h5":
        data, kwargs = read_hdf_map(filename)
    else:
        raise NotImplementedError(f"Unsupported filetype '.{format}'.")

    # if there are any kwargs specifying the size of the map
    # remove size kwargs from the read-in map's metadata
    # so that they overwrite the size
    overridden_size_kwargs = {}
    overriding_size_kwargs = {k: v for k, v in map_kwargs.items() if k in MAP_SIZE_KWARGS}
    if overriding_size_kwargs:
        for k in MAP_SIZE_KWARGS:
            if k in kwargs:
                overridden_size_kwargs[k] = kwargs.pop(k)

        logging.info(
            f"Passed map size kwargs {overriding_size_kwargs} will overwrite map size metadata {overridden_size_kwargs}"
        )

    kwargs.update(map_kwargs)
    logger.debug(f"Loading ProjectionMap with kwargs {kwargs}")

    return ProjectionMap(data=data, **kwargs)


def read_hdf_map(filename: str):
    with h5py.File(filename, "r") as f:
        kwargs = {}
        for field in f.keys():
            value = f[field][()]
            kwargs[field] = value if not isinstance(value, bytes) else value.decode()

        if "weight" in f.keys():
            kwargs["weight"] = f["weight"][:]

    data = kwargs.pop("data")

    return data, kwargs
