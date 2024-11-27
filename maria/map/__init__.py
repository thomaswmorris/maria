from __future__ import annotations

import os

import h5py
import numpy as np
from astropy.io import fits

from .base import Map
from .projected import ProjectedMap  # noqa

from matplotlib.colors import ListedColormap
import matplotlib as mpl

here, this_filename = os.path.split(__file__)

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/../plotting/Planck_Parchment_RGB.txt") / 255.0,
    name="cmb",
)
cmb_cmap.set_bad("white")
mpl.colormaps.register(cmb_cmap)


def load(filename: str, **kwargs) -> Map:
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

        metadata = {}
        for field in ["stokes", "nu", "t", "width", "center", "frame", "units"]:
            if field in f.keys():
                value = f[field][()]
                metadata[field] = (
                    value if not isinstance(value, bytes) else value.decode()
                )

        if "weight" in f.keys():
            metadata["weight"] = f["weight"][:]

    metadata.update(kwargs)

    return ProjectedMap(data=data, degrees=False, **metadata)


def read_fits(
    filename: str,
    index: int = 0,
    **map_kwargs,
) -> Map:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    hudl = fits.open(filename)

    indices_with_image = np.where([h.data is not None for h in hudl])[0]
    if len(indices_with_image) == 0:
        raise ValueError(f"FITS file '{filename}' has no images.")

    index = index or indices_with_image[0]

    map_data = hudl[index].data
    if map_data.ndim < 2:
        raise ValueError("Map should have at least 2 dimensions.")

    return ProjectedMap(data=map_data, **map_kwargs)
