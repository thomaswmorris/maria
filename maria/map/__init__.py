from __future__ import annotations

import os

import h5py
import numpy as np
from astropy.io import fits

from .map import Map

here, this_filename = os.path.split(__file__)


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
        for field in ["nu", "t", "width", "center", "frame", "units"]:
            value = f[field][()]
            metadata[field] = value if not isinstance(value, bytes) else value.decode()

        if "weight" in f.keys():
            metadata["weight"] = f["weight"][:]

    metadata.update(kwargs)

    return Map(data=data, degrees=False, **metadata)


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

    return Map(data=map_data, **map_kwargs)
