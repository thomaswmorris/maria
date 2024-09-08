import os

import astropy as ap

from .map import Map

here, this_filename = os.path.split(__file__)


def read_fits(
    filename: str,
    index: int = 0,
    **map_kwargs,
) -> Map:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    hudl = ap.io.fits.open(filename)

    map_data = hudl[index].data
    if map_data.ndim < 2:
        raise ValueError("Map should have at least 2 dimensions.")

    return Map(data=map_data, **map_kwargs)
