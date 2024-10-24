import os

import numpy as np
from astropy.io import fits

from .map import Map

here, this_filename = os.path.split(__file__)


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
