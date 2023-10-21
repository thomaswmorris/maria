import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pytz

from . import utils

here, this_filename = os.path.split(__file__)

POINTING_CONFIGS = utils.io.read_yaml(f"{here}/configs/pointings.yml")
POINTING_PARAMS = set()
for key, config in POINTING_CONFIGS.items():
    POINTING_PARAMS |= set(config.keys())


class UnsupportedPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(
            f"The site '{invalid_pointing}' is not in the database of default pointings. "
            f"Default pointings are:\n\n{sorted(list(POINTING_CONFIGS.keys()))}"
        )


def get_pointing_config(pointing_name, **kwargs):
    if pointing_name not in POINTING_CONFIGS.keys():
        raise UnsupportedPointingError(pointing_name)
    POINTING_CONFIG = POINTING_CONFIGS[pointing_name].copy()
    for k, v in kwargs.items():
        POINTING_CONFIG[k] = v
    return POINTING_CONFIG


def get_pointing(pointing_name, **kwargs):
    return Pointing(**get_pointing_config(pointing_name, **kwargs))


@dataclass
class Pointing:
    """
    A dataclass containing time-ordered pointing data.
    """

    description: str = ""
    utc_time: str = ""
    pointing_frame: str = "ra_dec"
    pointing_units: str = "degrees"
    scan_center: Tuple[float, float] = (4, 10.5)
    scan_radius: float = 1.0
    scan_period: float = 60.0
    scan_pattern: str = "daisy"
    scan_options: dict = field(default_factory=dict)
    start_time: float | str = "2022-02-10T06:00:00"
    integration_time: float = 60.0
    sample_rate: float = 20.0

    @staticmethod
    def validate_pointing_kwargs(kwargs):
        """
        Make sure that we have all the ingredients to produce the pointing data.
        """
        if ("end_time" not in kwargs.keys()) and (
            "integration_time" not in kwargs.keys()
        ):
            raise ValueError(
                'One of "end_time" or "integration_time" must be in the pointing kwargs.'
            )

    def __post_init__(self):
        if not hasattr(self, "start_time"):
            self.start_time = datetime.now().timestamp()
        self.start_datetime = utils.datetime_handler(self.start_time)
        self.end_datetime = self.start_datetime + timedelta(
            seconds=self.integration_time
        )

        self.time_min = self.start_datetime.timestamp()
        self.time_max = self.end_datetime.timestamp()
        self.dt = 1 / self.sample_rate

        if self.pointing_units == "degrees":
            self.scan_center_radians = np.radians(self.scan_center)
            self.scan_radius_radians = np.radians(self.scan_radius)
        else:
            self.scan_center_radians = np.array(self.scan_center)
            self.scan_radius_radians = np.array(self.scan_radius)

        self.time = np.arange(self.time_min, self.time_max, self.dt)
        self.n_time = len(self.time)

        time_ordered_pointing = utils.get_pointing(
            self.time,
            scan_center=self.scan_center_radians,  # a lon/lat in some frame
            pointing_frame=self.pointing_frame,  # the frame, one of "az_el", "ra_dec", "galactic"
            scan_radius=self.scan_radius_radians,
            scan_period=self.scan_period,
            scan_pattern="daisy",
        )

        if self.pointing_frame == "ra_dec":
            self.ra, self.dec = time_ordered_pointing
        elif self.pointing_frame == "az_el":
            self.az, self.el = time_ordered_pointing
        elif self.pointing_frame == "dx_dy":
            self.dx, self.dy = time_ordered_pointing

        self.utc_time = (
            datetime.fromtimestamp(self.time_min).astimezone(pytz.utc).ctime()
        )
