from __future__ import annotations

import os

import h5py
import matplotlib as mpl
import numpy as np
from astropy.io import fits
from matplotlib.colors import ListedColormap

from .base import Map
from .healpix import HEALPixMap  # noqa
from .projected import ProjectedMap  # noqa

here, this_filename = os.path.split(__file__)

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/../plotting/Planck_Parchment_RGB.txt") / 255.0,
    name="cmb",
)
cmb_cmap.set_bad("white")
mpl.colormaps.register(cmb_cmap)


VALID_MAP_KWARGS = ["stokes", "nu", "t", "center", "width", "height", "resolution", "frame", "units"]


def load(filename: str, **kwargs) -> Map:
    if "nu" in kwargs:
        kwargs["nu"] = kwargs["nu"]

    format = filename.split(".")[-1]
    if format == "fits":
        return read_fits(filename, **kwargs)
    if format == "h5":
        return read_hdf(filename, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported filetype '.{format}'.")


def read_hdf(filename: str, **kwargs) -> Map:
    with h5py.File(filename, "r") as f:
        data = f["data"][:]

        map_kwargs = {}
        for field in VALID_MAP_KWARGS:
            if field in f.keys():
                value = f[field][()]
                map_kwargs[field] = value if not isinstance(value, bytes) else value.decode()

        if "weight" in f.keys():
            map_kwargs["weight"] = f["weight"][:]

    map_kwargs.update(kwargs)

    return ProjectedMap(data=data, degrees=True, **map_kwargs)


def read_fits(
    filename: str,
    index: int = 0,
    **map_kwargs,
) -> Map:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    hdul = fits.open(filename)

    indices_with_image = np.where([h.data is not None for h in hdul])[0]
    if len(indices_with_image) == 0:
        raise ValueError(f"FITS file '{filename}' has no images.")

    index = index or indices_with_image[0]

    hdu = hdul[index]

    map_data = hdu.data
    if map_data.ndim < 2:
        raise ValueError("Map should have at least 2 dimensions.")

    for key in hdu.header.keys():
        if key.lower() in VALID_MAP_KWARGS:
            map_kwargs[key.lower()] = hdu.header[key]

    return ProjectedMap(data=map_data, degrees=True, **map_kwargs)
