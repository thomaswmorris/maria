from __future__ import annotations

import logging
import os
import re

import numpy as np
from astropy.io import fits
from astropy.io.fits.header import Header

from ...units import Quantity, parse_units

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")


FITS_TYPE_ALIASES = {
    "nu": ["NU", "FREQ"],
    "v": ["VRAD", "VELO"],
    "t": ["TIME"],
}

FITS_KWARG_ALIASES = {
    "units": [
        "BUNIT",
        "BUNITS",
    ],
    "nu": ["NU", "FREQ", "RESTFRQ", "RESTFREQ"],
    # "z": ["REDSHIFT"],
}

FITS_DEFAULT_UNITS = {
    "nu": "Hz",
    "v": "m/s",
    "t": "s",
    "eta": "deg",
    "xi": "deg",
}

FITS_FRAMES = {
    "ra/dec": {
        "xi": {"aliases": ["RA"], "parity": -1},
        "eta": {"aliases": ["DEC"], "parity": +1},
    },
    "galactic": {
        "xi": {"aliases": ["GLON"], "parity": -1},
        "eta": {"aliases": ["GLAT"], "parity": +1},
    },
}


def read_fits_map(path: str, index: int | None = None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with fits.open(path) as hdul:
        if index is None:
            indices_with_image = [index for index, h in enumerate(hdul) if h.data is not None]
            if len(indices_with_image) == 0:
                raise ValueError(f"Could not infer HDU index (FITS file '{path}' has no images).")
            index = indices_with_image[0]

        hdu = hdul[index]

        # FITS counts from the bottom, while normal people count from the top
        data = hdu.data  # [..., ::-1, :]
        header = hdu.header

    kwargs = parse_fits_map_header(header)

    return data, kwargs, header


def parse_fits_map_header(header):

    parity = {}
    axes = {}
    kwargs = {}

    center = [0, 0]
    for axis in range(header["NAXIS"]):
        parity = 1

        AXIS = axis + 1

        NAXIS = header[f"NAXIS{AXIS}"]
        CTYPE = header.get(f"CTYPE{AXIS}")

        if CTYPE is None:
            continue

        dim = None
        for dim_name, dim_aliases in FITS_TYPE_ALIASES.items():
            for alias in dim_aliases:
                if alias in CTYPE:
                    dim, default_units = dim_name, FITS_DEFAULT_UNITS[dim_name]
                    break

        for frame_name, frame_config in FITS_FRAMES.items():
            for dim_name, dim_config in frame_config.items():
                for alias in dim_config["aliases"]:
                    if alias in CTYPE:
                        dim, default_units = dim_name, FITS_DEFAULT_UNITS[dim_name]
                        kwargs["frame"] = frame_name
                        parity = dim_config["parity"]
                        break

        if dim is None:
            raise ValueError(f"Could not interpret CTYPE{AXIS} = {repr(CTYPE)}")

        CDELT = header.get(f"CDELT{AXIS}")
        CRVAL = header.get(f"CRVAL{AXIS}")
        CRPIX = header.get(f"CRPIX{AXIS}")
        CUNIT = header.get(f"CUNIT{AXIS}", default_units)

        logger.debug(f"Found CTYPE{AXIS}: {repr(CTYPE)}")
        logger.debug(f"Found CDELT{AXIS}: {CDELT}")
        logger.debug(f"Found CRVAL{AXIS}: {CRVAL}")
        logger.debug(f"Found CRPIX{AXIS}: {CRPIX}")
        logger.debug(f"Found CUNIT{AXIS}: {repr(CUNIT)}")
        logger.debug(f"Interpreting axis {AXIS} as dimension '{dim}'")

        axis_values = CDELT * np.arange(NAXIS, dtype=float)
        axis_values -= np.interp(CRPIX, np.arange(NAXIS), axis_values)
        axis_values *= parity
        logger.debug(f"Applied parity {parity} for dimension '{dim}'")

        if dim not in ["xi", "eta"]:
            axis_values += CRVAL
        else:
            center[["xi", "eta"].index(dim)] = CRVAL

        axes[dim] = Quantity(axis_values, CUNIT)

    for key, value in header.items():
        for kwarg, aliases in FITS_KWARG_ALIASES.items():
            for alias in aliases:
                if key == alias:
                    logger.debug(f"Using {alias}: {value} for kwarg '{kwarg}'")
                    if kwarg in ["nu", "z"]:
                        axes[kwarg] = Quantity([value], FITS_DEFAULT_UNITS[kwarg])
                    else:
                        kwargs[kwarg] = value
                    break

        beam = [0, 0, 0]
        beam_keys = [["BMAJ", "BMAJOR"], ["BMIN", "BMINOR"], ["BPA"]]
        for beam_index, aliases in enumerate(beam_keys):
            for alias in aliases:
                if alias in header:
                    beam[beam_index] = header[alias]
                    logger.debug(f"Using {alias}: {value} as beam parameter")
                    break

    return {"center": center, "beam": beam, **kwargs, **axes}
