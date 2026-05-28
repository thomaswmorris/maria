from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping

import astropy as ap
import h5py
import pandas as pd
import yaml

from .caching import *  # noqa
from .coords import *  # noqa
from .fits import *  # noqa
from .logging import *  # noqa
from .parsing import *  # noqa
from .repr import *  # noqa

logger = logging.getLogger("maria")


def read_yaml(path: str):
    """
    Return a YAML file as a dict
    """
    res = yaml.safe_load(pathlib.Path(path).read_text())
    return res if res is not None else {}
