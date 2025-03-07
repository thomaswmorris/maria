from __future__ import annotations

import glob
import os
from collections.abc import Mapping

import pandas as pd

from ..utils import flatten_config, read_yaml
from .band import Band, get_band  # noqa
from .band_list import BandList  # noqa

here, this_filename = os.path.split(__file__)

BAND_FIELD_FORMATS = pd.read_csv(f"{here}/format.csv", index_col=0)

BAND_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    tag = os.path.split(path)[1].split(".")[0]
    BAND_CONFIGS[tag] = read_yaml(path)

BAND_CONFIGS = flatten_config(BAND_CONFIGS)

band_data = pd.DataFrame(BAND_CONFIGS).T.sort_index()
all_bands = list(band_data.index)


def validate_band_config(band):
    if "passband" not in band:
        if any([key not in band for key in ["center", "width"]]):
            raise ValueError("The band's center and width must be specified!")


def parse_bands(bands):
    """
    Take in a flexible format of a band specification, and return a list of bands.
    """
    band_list = []

    if isinstance(bands, list):
        for band in bands:
            if isinstance(band, Band):
                band_list.append(band)
            elif isinstance(band, str):
                band_list.append(get_band(band_name=band))
            else:
                raise TypeError("'band' must be either a Band or a string.")
        return band_list

    elif isinstance(bands, Mapping):
        for band_name, band in bands.items():
            if isinstance(band, Band):
                band_list.append(band)
            elif isinstance(band, Mapping):
                band_list.append(Band(name=band_name, **band))

    else:
        raise TypeError("'bands' must be either a list or a mapping.")

    return band_list
