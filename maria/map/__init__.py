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
from ..io import fetch
from ..units import Quantity, parse_units
from .base import VALID_MAP_QUANTITIES, Map, concatenate  # noqa
from .healpix import HEALPixMap  # noqa
from .projection import ProjectionMap  # noqa

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
        "aliases": ["VRAD"],
        "default_units": "m/s",
    },
}


def get(name: str, **kwargs):
    return load(fetch(name), **kwargs)


def load(filename: str, format: str = "auto", **map_kwargs) -> Map:
    if format == "auto":
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
    logger.debug(f"Loading ProjectionMap with metadata {metadata}")

    return ProjectionMap(data=data, **metadata)


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


def read_fits(filename: str, index: int | None = None):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with fits.open(filename) as hdul:
        if index is None:
            indices_with_image = [index for index, h in enumerate(hdul) if h.data is not None]
            if len(indices_with_image) == 0:
                raise ValueError(f"Could not infer HDU index (FITS file '{filename}' has no images).")
            index = indices_with_image[0]

        hdu = hdul[index]
        header_dict = dict(hdu.header)

        # FITS counts from the bottom, while normal people count from the top
        data = hdu.data[::-1, :]
        if data.ndim < 2:
            raise ValueError("Map should have at least 2 dimensions.")

        frame = {}
        center = {}
        metadata = {}
        for kwarg, aliases in FITS_KEYWORD_ALIASES.items():
            for alias in aliases:
                if alias in header_dict:
                    metadata[kwarg] = header_dict[alias]
                    logger.debug(f"Found value {alias}={header_dict[alias]} for map kwarg '{kwarg}'")
                    break

        for axis in range(header_dict["NAXIS"]):
            AXIS = axis + 1
            CTYPE = header_dict.get(f"CTYPE{AXIS}", None)
            if CTYPE is None:
                logger.debug(f"Could not find CTYPE for AXIS{AXIS}")
                continue

            if axis < 2:
                dim = "xy"[axis]
                match = re.compile(r"(.+?)-*([A-Z]{3})").match(CTYPE)
                if match:
                    coord, proj = match.groups()
                    frame[dim] = coord
                    logger.debug(f"Using CTYPE{AXIS} = {CTYPE} for dim {dim}")
                else:
                    raise ValueError(f"Invalid CTYPE {CTYPE}")

                units = header_dict.get(f"CUNIT{AXIS}", "deg")

                if f"CRVAL{AXIS}" in header_dict:
                    value = header_dict[f"CRVAL{AXIS}"]

                    center[dim] = Quantity(
                        value, units
                    ).deg  # negative because from inside the sphere, latitude goes right to left

                if f"CDELT{AXIS}" in header_dict:
                    value = header_dict[f"CDELT{AXIS}"]

                    metadata[f"{dim}_res"] = Quantity(
                        value, units
                    ).deg  # negative because from inside the sphere, latitude goes right to left

            else:
                for axis, axis_mapping in AXIS_MAPPING.items():
                    if CTYPE.upper() in axis_mapping["aliases"]:
                        logger.debug(f"Using CAXIS{AXIS} for kwarg '{axis}'")
                        axis_units = parse_units(axis_mapping["default_units"])
                        if f"CUNIT{AXIS}" in header_dict:
                            try:
                                axis_units = parse_units(header_dict[f"CUNIT{AXIS}"])
                            except Exception as error:
                                logger.warning(
                                    f"Could not infer units for AXIS3, assuming units of {axis_mapping['default_units']}"
                                )

                        CDELT = header_dict[f"CDELT{AXIS}"]
                        CRVAL = header_dict[f"CRVAL{AXIS}"]
                        CRPIX = header_dict[f"CRPIX{AXIS}"]

                        axis_values = CDELT * np.arange(header_dict[f"NAXIS{AXIS}"])
                        axis_values += CRVAL - axis_values[int(CRPIX)]
                        metadata[axis] = Quantity(axis_values, axis_units["units"])

                        break

        beam = [0, 0, 0]
        beam_keys = [["BMAJ", "BMAJOR"], ["BMIN", "BMINOR"], ["BPA"]]
        for index in range(3):
            for key in beam_keys[index]:
                if key in header_dict:
                    value = header_dict[key]
                    beam[index] = Quantity(value, units).deg
                    logger.debug(f"Found beam param {key} = {value}")

        for fits_key in header_dict:
            for maria_key in FITS_KEYWORD_ALIASES:
                for mapped_key in FITS_KEYWORD_ALIASES[maria_key]:
                    if mapped_key == fits_key:
                        value = hdu.header[fits_key]
                        logger.debug(f"Using {fits_key} = {value} for kwarg {maria_key}")
                        metadata[maria_key] = value

        # logger.warning("Could not infer spatial dimensions from FITS header; specify map dimensions manually")

    for frame_name, f in frames.items():
        if (f["FITS_phi"], f["FITS_theta"]) == (frame["x"], frame["y"]):
            metadata["frame"] = frame_name
            logger.debug(f"Using frame '{frame_name}'")

    if "frame" not in metadata:
        metadata["frame"] = "ra/dec"
        logger.warning(f"Could not infer coordinate system from FITS header; assuming frame 'ra/dec'")

    metadata["center"] = (center.get("x"), center.get("y"))
    metadata["degrees"] = "deg" in units
    metadata["beam"] = beam

    if metadata["frame"] in ["ra/dec", "galactic"]:
        data = data[..., ::-1]

    return data, metadata
