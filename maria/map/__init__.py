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

from .base import Map
from .healpix import HEALPixMap  # noqa
from .projected import ProjectedMap  # noqa

from ..units import Quantity
from ..coords import frames

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
    
    with fits.open(filename) as hdul:

        indices_with_image = [index for index, h in enumerate(hdul) if h.data is not None]
        if len(indices_with_image) == 0:
            raise ValueError(f"FITS file '{filename}' has no images.")

        index = index or indices_with_image[0]

        hdu = hdul[index]
        header_dict = dict(hdu.header)

        # FITS counts from the bottom, while normal people count from the top
        data = hdu.data[..., ::-1, :]
        if data.ndim < 2:
            raise ValueError("Map should have at least 2 dimensions.")

        frame = {}
        center = {}
        metadata = {}
        for axis, dim in enumerate("xy"):
            
            # metadata[f"{dim}_res"] = header_dict.get(f"CDELT{axis+1}", None)
            center[dim] = header_dict.get(f"CRVAL{axis+1}", None)
            units = header_dict.get(f"CUNIT{axis+1}", "deg").lower().strip()
            logger.debug(f"Found units '{units}' for dim {dim}")
            
            if f"CDELT{axis+1}" in header_dict:
                value = header_dict[f"CDELT{axis+1}"]
                logger.debug(f"Found CDELT {value} for dim {dim}")
                metadata[f"cdelt{axis+1}"] = -Quantity(value, units).deg # negative because from inside the sphere, latitude goes right to left

            CTYPE = header_dict.get(f"CTYPE{axis+1}", "RA---SIN")
            match = re.compile(r"(.+?)-*(SIN|TAN)").match(CTYPE)
            if match:
                coord, proj = match.groups()
                frame[dim] = coord
                logger.debug(f"Found CTYPE '{coord}' for dim {dim}")
            else:
                frame[dim] = None
                logger.debug(f"Could not find CTYPE for dim {dim}")

        for fits_key in header_dict:
            for maria_key in FITS_KEYWORD_MAPPING:
                for mapped_key in FITS_KEYWORD_MAPPING[maria_key]:
                    if mapped_key == fits_key:
                        logger.debug(f'Using FITS keyword "{fits_key}" for kwarg "{maria_key}"')
                        metadata[maria_key] = hdu.header[fits_key]
            
        # logger.warning("Could not infer spatial dimensions from FITS header; specify map dimensions manually")

    for frame_name, f in frames.items():
        if (f["FITS_phi"], f["FITS_theta"]) == (frame["x"], frame["y"]):
            metadata["frame"] = frame_name
            logger.debug(f"Using frame '{frame_name}'")
            
    if "frame" not in metadata:
        metadata["frame"] = "ra_dec"
        logger.warning(f"Could not infer coordinate system from FITS header; assuming frame 'ra_dec'")

    if center["x"] and center["y"]:
        metadata["center"] = (center["x"], center["y"])

    return data, metadata
