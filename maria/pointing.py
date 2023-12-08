import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import pytz

from . import utils

here, this_filename = os.path.split(__file__)

POINTING_CONFIGS = utils.io.read_yaml(f"{here}/configs/pointings.yml")
POINTING_PARAMS = set()
for key, config in POINTING_CONFIGS.items():
    POINTING_PARAMS |= set(config.keys())
supported_pointings_table = pd.DataFrame(POINTING_CONFIGS).T
all_pointings = list(supported_pointings_table.index.values)


class UnsupportedPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(
            f"The site '{invalid_pointing}' is not in the database of default pointings. "
            f"Default pointings are:\n\n{supported_pointings_table.to_string()}"
        )


def get_pointing_config(pointing_name="default", **kwargs):
    if pointing_name not in POINTING_CONFIGS.keys():
        raise UnsupportedPointingError(pointing_name)
    POINTING_CONFIG = POINTING_CONFIGS[pointing_name].copy()
    for k, v in kwargs.items():
        POINTING_CONFIG[k] = v
    return POINTING_CONFIG


def get_pointing(pointing_name="default", **kwargs):
    return Pointing(**get_pointing_config(pointing_name, **kwargs))


def get_offsets(scan_pattern, integration_time, sample_rate, **scan_options):
    """
    Returns x and y offsets
    """

    if scan_pattern == "stare":
        return utils.pointing.get_stare_offsets(
            integration_time=integration_time, sample_rate=sample_rate, **scan_options
        )

    if scan_pattern == "daisy":
        return utils.pointing.get_daisy_offsets_constant_speed(
            integration_time=integration_time, sample_rate=sample_rate, **scan_options
        )

    elif scan_pattern == "back_and_forth":
        return utils.pointing.get_back_and_forth_offsets(
            integration_time=integration_time, sample_rate=sample_rate, **scan_options
        )

    else:
        raise ValueError(f"'{scan_pattern}' is not a valid scan pattern.")


@dataclass
class Pointing:
    """
    A dataclass containing time-ordered pointing data.
    """

    description: str = ""
    start_time: float | str = "2022-02-10T06:00:00"
    integration_time: float = 60.0
    sample_rate: float = 20.0
    pointing_frame: str = "ra_dec"
    pointing_units: str = "degrees"
    scan_center: Tuple[float, float] = (4, 10.5)
    scan_pattern: str = "daisy_miss_center"
    scan_options: dict = field(default_factory=dict)

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
        assert self.sample_rate > 0

        self.scan_center = tuple(self.scan_center)

        for k, v in POINTING_CONFIGS[self.scan_pattern]["scan_options"].items():
            if k not in self.scan_options.keys():
                self.scan_options[k] = v

        if not hasattr(self, "start_time"):
            self.start_time = datetime.now().timestamp()
        self.start_datetime = utils.io.datetime_handler(self.start_time)
        self.end_datetime = self.start_datetime + timedelta(
            seconds=self.integration_time
        )

        self.time_min = self.start_datetime.timestamp()
        self.time_max = self.end_datetime.timestamp()
        self.dt = 1 / self.sample_rate

        self.time = np.arange(self.time_min, self.time_max, self.dt)
        self.n_time = len(self.time)

        # this is in pointing_units
        x_scan_offsets, y_scan_offsets = getattr(utils.pointing, self.scan_pattern)(
            integration_time=self.integration_time,
            sample_rate=self.sample_rate,
            **self.scan_options,
        )

        if self.pointing_units == "degrees":
            self.scan_center_radians = (
                np.radians(self.scan_center[0]),
                np.radians(self.scan_center[1]),
            )
            x_scan_offsets_radians = np.radians(x_scan_offsets)
            y_scan_offsets_radians = np.radians(y_scan_offsets)
        elif self.pointing_units == "radians":
            self.scan_center_radians = self.scan_center
            x_scan_offsets_radians = x_scan_offsets
            y_scan_offsets_radians = y_scan_offsets
        else:
            raise ValueError('pointing units must be either "degrees" or "radians"')

        theta, phi = utils.coords.xy_to_lonlat(
            x_scan_offsets_radians, y_scan_offsets_radians, *self.scan_center_radians
        )

        if self.pointing_frame == "ra_dec":
            self.ra, self.dec = theta, phi
        elif self.pointing_frame == "az_el":
            self.az, self.el = theta, phi
        elif self.pointing_frame == "dx_dy":
            self.dx, self.dy = theta, phi
        else:
            raise ValueError("Not a valid pointing frame!")

        self.utc_time = (
            datetime.fromtimestamp(self.time_min).astimezone(pytz.utc).ctime()
        )
