from __future__ import annotations

import logging
import os

import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.colors import ListedColormap

from .base import Map
from .healpix import HEALPixMap  # noqa
from .projected import ProjectedMap  # noqa

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

MAP_SIZE_KWARGS = ["width", "height", "x_res", "y_res", "resolution"]
VALID_MAP_KWARGS = ["stokes", "nu", "t", "center", "frame", "units", *MAP_SIZE_KWARGS]

FITS_KEYWORD_MAPPING = {
    "frame": ["FRAME"],
    "x_res": ["CDELT1"],
    "y_res": ["CDELT2"],
    "units": ["BUNIT"],
    "nu": ["RESTFREQ"],
}


def load(filename: str, **map_kwargs) -> Map:
    format = filename.split(".")[-1].lower()
    if format == "fits":
        data, metadata = read_fits(filename)
    elif format == "h5":
        data, metadata = read_hdf(filename)
    else:
        raise NotImplementedError(f"Unsupported filetype '.{format}'.")

    # if there are any kwargs specifying the size of the map,
    # then remove size kwargs from the read-in map's metadata.
    size_kwargs = [k for k in map_kwargs if k in MAP_SIZE_KWARGS]

    if size_kwargs:
        metadata = {k: v for k, v in metadata.items() if k not in MAP_SIZE_KWARGS}

    metadata.update(map_kwargs)

    logger.debug(f"Loading ProjectedMap with metadata {metadata}")

    return ProjectedMap(data=data, degrees=True, **metadata)


def read_hdf(filename: str):
    with h5py.File(filename, "r") as f:
        data = f["data"][:]

        metadata = {}
        for field in VALID_MAP_KWARGS:
            if field in f.keys():
                value = f[field][()]
                metadata[field] = value if not isinstance(value, bytes) else value.decode()

        if "weight" in f.keys():
            metadata["weight"] = f["weight"][:]

    return data, metadata


def read_fits(filename: str, index: int = 0) -> Map:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    hdul = fits.open(filename)

    indices_with_image = np.where([h.data is not None for h in hdul])[0]
    if len(indices_with_image) == 0:
        raise ValueError(f"FITS file '{filename}' has no images.")

    index = index or indices_with_image[0]

    hdu = hdul[index]

    data = hdu.data
    if data.ndim < 2:
        raise ValueError("Map should have at least 2 dimensions.")

    metadata = {}
    for fits_key in hdu.header.keys():
        for maria_key in FITS_KEYWORD_MAPPING:
            for mapped_key in FITS_KEYWORD_MAPPING[maria_key]:
                if mapped_key == fits_key:
                    logger.debug(f'Using FITS keyword "{fits_key}" for metadata key "{maria_key}"')
                    metadata[maria_key] = hdu.header[fits_key]

    return data, metadata
