from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import arrow
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy as sp
from arrow import Arrow
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt

from .. import coords
from ..coords import Coordinates, frames
from ..io import DEFAULT_TIME_FORMAT, repr_lat_lon, repr_phi_theta
from ..units import Quantity
from ..utils import compute_diameter, read_yaml
from .patterns import get_scan_pattern_generator, scan_patterns
from .plan import Plan
from .plan_list import PlanList  # noqa
from .planner import Planner  # noqa

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

all_patterns = list(scan_patterns.index.values)

MAX_VELOCITY_WARN = 10  # in deg/s
MAX_ACCELERATION_WARN = 10  # in deg/s

MIN_ELEVATION_WARN = 20  # in deg
MIN_ELEVATION_ERROR = 10  # in deg

here, this_filename = os.path.split(__file__)

PLAN_CONFIGS = {}
for plans_path in Path(f"{here}/plans").glob("*.yml"):
    PLAN_CONFIGS.update(read_yaml(plans_path))

plan_params = set()
for key, config in PLAN_CONFIGS.items():
    plan_params |= set(config.keys())
plan_data = pd.DataFrame(PLAN_CONFIGS).T
all_plans = list(plan_data.index.values)


class UnsupportedPlanError(Exception):
    def __init__(self, invalid_plan):
        super().__init__(
            f"The plan '{invalid_plan}' is not a supported plan. Supported plans are: \n\n{plan_data.to_string()}",
        )


def get_plan_config(plan_name="ten_second_zenith_stare", **kwargs):
    if plan_name not in PLAN_CONFIGS.keys():
        raise UnsupportedPlanError(plan_name)
    plan_config = PLAN_CONFIGS[plan_name].copy()
    for k, v in kwargs.items():
        plan_config[k] = v
    return plan_config


def get_plan(plan_name="ten_second_zenith_stare", **kwargs):
    plan_config = get_plan_config(plan_name, **kwargs)
    return Plan.generate(**plan_config)


PLAN_FIELDS = {
    "start_time": Union[float, str, Arrow],
    "duration": float,
    "sample_rate": float,
    "frame": str,
    "degrees": bool,
    "scan_center": float,
    "scan_pattern": float,
    "scan_options": Mapping,
}


def validate_pointing_kwargs(kwargs):
    """
    Make sure that we have all the ingredients to produce the plan data.
    """
    if ("end_time" not in kwargs.keys()) and ("duration" not in kwargs.keys()):
        raise ValueError(
            """One of 'end_time' or 'duration' must be in the plan kwargs.""",
        )
