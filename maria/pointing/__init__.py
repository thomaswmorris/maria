import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz

from .. import coords, utils
from . import patterns

MAX_VELOCITY_WARN = np.radians(10)  # in deg/s
MAX_ACCELERATION_WARN = np.radians(10)  # in deg/s

here, this_filename = os.path.split(__file__)

pointing_configs = utils.io.read_yaml(f"{here}/pointings.yml")
pointing_params = set()
for key, config in pointing_configs.items():
    pointing_params |= set(config.keys())
pointing_data = pd.DataFrame(pointing_configs).T
all_pointings = list(pointing_data.index.values)


class UnsupportedPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(
            f"""The site '{invalid_pointing}' is not in the database of default pointings."""
            f"""Default pointings are:\n\n{pointing_data.to_string()}"""
        )


def get_pointing_config(scan_pattern="stare", **kwargs):
    if scan_pattern not in pointing_configs.keys():
        raise UnsupportedPointingError(scan_pattern)
    pointing_config = pointing_configs[scan_pattern].copy()
    for k, v in kwargs.items():
        pointing_config[k] = v
    return pointing_config


def get_pointing(scan_pattern="stare", **kwargs):
    pointing_config = get_pointing_config(scan_pattern, **kwargs)
    return Pointing(**pointing_config)


@dataclass
class Pointing:
    """
    A dataclass containing time-ordered pointing data.
    """

    pointing_description: str = ""
    start_time: str = "2022-02-10T06:00:00"
    integration_time: float = 60.0
    sample_rate: float = 20.0
    pointing_frame: str = "ra_dec"
    degrees: bool = True
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
                """One of 'end_time' or 'integration_time' must be in the pointing kwargs."""
            )

    def __post_init__(self):
        assert self.sample_rate > 0

        self.scan_center = tuple(self.scan_center)

        for k, v in pointing_configs[self.scan_pattern]["scan_options"].items():
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
        x_scan_offsets, y_scan_offsets = getattr(patterns, self.scan_pattern)(
            integration_time=self.integration_time,
            sample_rate=self.sample_rate,
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

        scan_velocity_radians = np.gradient(
            self.scan_offsets_radians, axis=1, edge_order=0
        ) / np.gradient(self.time)
        scan_acceleration_radians = np.gradient(
            scan_velocity_radians, axis=1, edge_order=0
        ) / np.gradient(self.time)

        self.max_vel = np.sqrt(np.sum(scan_velocity_radians**2, axis=0)).max()
        self.max_acc = np.sqrt(np.sum(scan_acceleration_radians**2, axis=0)).max()

        if self.max_vel > MAX_VELOCITY_WARN:
            warnings.warn(
                (
                    f"The maximum velocity of the boresight ({np.degrees(self.max_vel):.01f} deg/s) is "
                    "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
                ),
                stacklevel=2,
            )

        if self.max_acc > MAX_ACCELERATION_WARN:
            warnings.warn(
                (
                    f"The maximum acceleration of the boresight ({np.degrees(self.max_acc):.01f} deg/s^2) is "
                    "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
                ),
                stacklevel=2,
            )

        self.phi, self.theta = coords.dx_dy_to_phi_theta(
            *self.scan_offsets_radians, *self.scan_center_radians
        )
        if self.pointing_frame == "ra_dec":
            self.ra, self.dec = self.phi, self.theta
        elif self.pointing_frame == "az_el":
            self.az, self.el = self.phi, self.theta
        else:
            raise ValueError("Not a valid pointing frame!")

        self.utc_time = (
            datetime.fromtimestamp(self.time_min).astimezone(pytz.utc).ctime()
        )

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        max_scan_offset = self.scan_offsets_radians.ptp(axis=1).max()

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
            f"""{coords.frames[self.pointing_frame]['phi_name']} = {center_phi} {pointing_units}"""
            f"""{coords.frames[self.pointing_frame]['theta_name']} = {center_theta} {pointing_units}"""
        )

        ax.plot(dx, dy, lw=5e-1)
        ax.scatter(0, 0, c="r", marker="x", label=label)
        ax.set_xlabel(rf"$\Delta \, \theta_x$ [{units}]")
        ax.set_ylabel(rf"$\Delta \, \theta_y$ [{units}]")
        ax.legend()
