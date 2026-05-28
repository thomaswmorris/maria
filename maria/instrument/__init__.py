from __future__ import annotations

import glob
import os
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from ..array import Array, ArrayList, get_array_config  # noqa
from ..band import BAND_CONFIGS, Band, BandList, parse_band  # noqa
from ..beam import compute_angular_fwhm
from ..io import flatten_config, read_yaml
from ..units import Quantity
from ..utils import HEX_CODE_LIST
from .instrument import Instrument  # noqa

here, this_filename = os.path.split(__file__)

INSTRUMENT_CONFIGS = {}
for path in glob.glob(f"{here}/configs/*.yml"):
    key = os.path.split(path)[1].split(".")[0]
    INSTRUMENT_CONFIGS[key] = read_yaml(path)
INSTRUMENT_CONFIGS = flatten_config(INSTRUMENT_CONFIGS)

# better formatting for pandas dataframes
# pd.set_eng_float_format()

for name, config in INSTRUMENT_CONFIGS.items():
    config["aliases"] = config.get("aliases", [])
    config["aliases"].append(name.lower())

INSTRUMENT_DISPLAY_COLUMNS = [
    "description",
    # "field_of_view",
    # "primary_size",
    # "bands",
]


def get_instrument(name=None, **kwargs):
    config = get_instrument_config(name) if name else {}
    config.update(kwargs)
    return Instrument.from_config(config)


def get_instrument_config(name):
    for v in INSTRUMENT_CONFIGS.values():
        if name.lower() in v["aliases"]:
            return v.copy()
    raise KeyError(f"'{name}' is not a valid array name.")


instrument_data = pd.DataFrame(INSTRUMENT_CONFIGS).reindex(INSTRUMENT_DISPLAY_COLUMNS).T

# for instrument_name, config in INSTRUMENT_CONFIGS.items():
#     instrument = get_instrument(instrument_name)
#     f_list = sorted(np.unique([band.center for band in instrument.dets.bands]))
#     instrument_data.at[instrument_name, "f [GHz]"] = "/".join([str(f) for f in f_list])
#     instrument_data.at[instrument_name, "n"] = instrument.dets.n

all_instruments = list(instrument_data.index)
test_instruments = ["test/1deg"]


class InvalidInstrumentError(Exception):
    def __init__(self, invalid_instrument):
        super().__init__(
            f"The instrument '{invalid_instrument}' is not supported. "
            f"Supported instruments are:\n\n{instrument_data.__repr__()}",
        )
