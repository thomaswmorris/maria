import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

from .. import coords
from ..io import datetime_handler, read_yaml
from .patterns import get_pattern_generator, patterns

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

all_patterns = list(patterns.index.values)

MAX_VELOCITY_WARN = 10  # in deg/s
MAX_ACCELERATION_WARN = 10  # in deg/s

MIN_ELEVATION_WARN = 20  # in deg
MIN_ELEVATION_ERROR = 10  # in deg

plan_configs = read_yaml(f"{here}/plans.yml")
plan_params = set()
for key, config in plan_configs.items():
    plan_params |= set(config.keys())
plan_data = pd.DataFrame(plan_configs).T
all_plans = list(plan_data.index.values)


class UnsupportedPlanError(Exception):
    def __init__(self, invalid_plan):
        super().__init__(
            f"""The plan '{invalid_plan}' is not a supported plan. """
            f"""Supported plans are:\n\n{plan_data.to_string()}"""
        )


def get_plan_config(plan_name="one_minute_zenith_stare", **kwargs):
    if plan_name not in plan_configs.keys():
        raise UnsupportedPlanError(plan_name)
    plan_config = plan_configs[plan_name].copy()
    for k, v in kwargs.items():
        plan_config[k] = v
    return plan_config


def get_plan(plan_name="one_minute_zenith_stare", **kwargs):
    plan_config = get_plan_config(plan_name, **kwargs)
    return Plan(**plan_config)


@dataclass
class Plan:
    """
    A dataclass containing time-ordered plan data.
    """

    description: str = ""
    start_time: Union[str, int] = "2022-02-10T06:00:00"
    duration: float = 60.0
    sample_rate: float = 20.0
    frame: str = "ra_dec"
    degrees: bool = True
    scan_center: Tuple[float, float] = (4, 10.5)
    scan_pattern: str = "daisy_miss_center"
    scan_options: dict = field(default_factory=dict)

    @staticmethod
    def validate_pointing_kwargs(kwargs):
        """
        Make sure that we have all the ingredients to produce the plan data.
        """
        if ("end_time" not in kwargs.keys()) and ("duration" not in kwargs.keys()):
            raise ValueError(
                """One of 'end_time' or 'duration' must be in the plan kwargs."""
            )

    def __post_init__(self):
        if not self.sample_rate > 0:
            raise ValueError("Parameter 'sample_rate' must be greater than zero!")

        self.scan_center = tuple(np.array(self.scan_center))

        # for k, v in plan_configs[self.scan_pattern]["scan_options"].items():
        #     if k not in self.scan_options.keys():
        #         self.scan_options[k] = v

        if not hasattr(self, "start_time"):
            self.start_time = datetime.now().timestamp()
        self.start_datetime = datetime_handler(self.start_time)
        self.end_datetime = self.start_datetime + timedelta(seconds=self.duration)

        self.time_min = self.start_datetime.timestamp()
        self.time_max = self.end_datetime.timestamp()
        self.dt = 1 / self.sample_rate

        self.time = np.arange(self.time_min, self.time_max, self.dt)
        self.n_time = len(self.time)

        # convert radius to width / height
        if "width" in self.scan_options:
            self.scan_options["radius"] = 0.5 * self.scan_options.pop("width")

        # this is in pointing_units
        x_scan_offsets, y_scan_offsets = get_pattern_generator(self.scan_pattern)(
            self.time,
            **self.scan_options,
        )

        if self.degrees:
            self.scan_center_radians = (
                np.radians(self.scan_center[0]),
                np.radians(self.scan_center[1]),
            )
            x_scan_offsets_radians = np.radians(x_scan_offsets)
            y_scan_offsets_radians = np.radians(y_scan_offsets)
        else:
            self.scan_center_radians = self.scan_center
            x_scan_offsets_radians = x_scan_offsets
            y_scan_offsets_radians = y_scan_offsets

        self.scan_offsets_radians = np.c_[
            x_scan_offsets_radians, y_scan_offsets_radians
        ].T

        # add 0.1 arcseconds of jitter
        self.scan_offsets_radians += np.radians(0.1 / 3600) * np.random.standard_normal(
            size=self.scan_offsets_radians.shape
        )

        scan_velocity_radians = np.gradient(
            self.scan_offsets_radians, axis=1, edge_order=0
        ) / np.gradient(self.time)
        scan_acceleration_radians = np.gradient(
            scan_velocity_radians, axis=1, edge_order=0
        ) / np.gradient(self.time)

        self.max_vel = np.sqrt(np.sum(scan_velocity_radians**2, axis=0)).max()
        self.max_acc = np.sqrt(np.sum(scan_acceleration_radians**2, axis=0)).max()

        if self.max_vel > MAX_VELOCITY_WARN:
            logger.warning(
                (
                    f"The maximum velocity of the boresight ({np.degrees(self.max_vel):.01f} deg/s) is "
                    "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
                ),
                stacklevel=2,
            )

        if self.max_acc > MAX_ACCELERATION_WARN:
            logger.warning(
                (
                    f"The maximum acceleration of the boresight ({np.degrees(self.max_acc):.01f} deg/s^2) is "
                    "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
                ),
                stacklevel=2,
            )

        self.phi, self.theta = coords.dx_dy_to_phi_theta(
            *self.scan_offsets_radians, *self.scan_center_radians
        )
        if self.frame == "ra_dec":
            self.ra, self.dec = self.phi, self.theta
        elif self.frame == "az_el":
            self.az, self.el = self.phi, self.theta
        else:
            raise ValueError("Not a valid pointing frame!")

        self.utc_time = (
            datetime.fromtimestamp(self.time_min).astimezone(pytz.utc).ctime()
        )

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        max_scan_offset = np.ptp(self.scan_offsets_radians, axis=1).max()

        if max_scan_offset < np.radians(0.5 / 60):
            dx, dy = 3600 * np.degrees(self.scan_offsets_radians)
            units = "arcsec."
        elif max_scan_offset < np.radians(0.5):
            dx, dy = 60 * np.degrees(self.scan_offsets_radians)
            units = "arcmin."
        else:
            dx, dy = np.degrees(self.scan_offsets_radians)
            units = "deg."

        center_phi, center_theta = self.scan_center

        pointing_units = "deg." if self.degrees else "rad."

        label = (
            f"""{coords.frames[self.frame]['phi_name']} = {center_phi} {pointing_units}"""
            f"""{coords.frames[self.frame]['theta_name']} = {center_theta} {pointing_units}"""
        )

        ax.plot(dx, dy, lw=5e-1)
        ax.scatter(0, 0, c="r", marker="x", label=label)
        ax.set_xlabel(rf"$\Delta \, \theta_x$ [{units}]")
        ax.set_ylabel(rf"$\Delta \, \theta_y$ [{units}]")
        ax.legend()
