from __future__ import annotations

import glob
import os

import pandas as pd

from ..utils import flatten_config, read_yaml
from .band import Band, get_band, parse_band  # noqa
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
