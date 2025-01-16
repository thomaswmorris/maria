from __future__ import annotations

import arrow
import logging
import os

from collections.abc import Mapping
from typing import Union
from pathlib import Path
from arrow import Arrow

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

from .. import coords
from ..units import Angle
from ..utils import read_yaml
from .patterns import get_pattern_generator, patterns

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

all_patterns = list(patterns.index.values)

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
            f"The plan '{invalid_plan}' is not a supported plan. "
            f"Supported plans are: \n\n{plan_data.to_string()}",
        )


def get_plan_config(plan_name="one_minute_zenith_stare", **kwargs):
    if plan_name not in PLAN_CONFIGS.keys():
        raise UnsupportedPlanError(plan_name)
    plan_config = PLAN_CONFIGS[plan_name].copy()
    for k, v in kwargs.items():
        plan_config[k] = v
    return plan_config


def get_plan(plan_name="one_minute_zenith_stare", **kwargs):
    plan_config = get_plan_config(plan_name, **kwargs)
    return Plan(**plan_config)


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


class Plan:
    """
    A dataclass containing time-ordered plan data.
    """

    def __repr__(self):
        parts = []
        frame = coords.frames[self.frame]
        center_degrees = self.scan_center.degrees

        parts.append(f"start_time={self.start_time.format()}")
        parts.append(
            f"center[{frame['phi']}, {frame['theta']}]=({center_degrees[0]:.02f}°, {center_degrees[1]:.02f}°)"
        )
        parts.append(f"pattern={self.scan_pattern}")
        parts.append(f"pattern_kwargs={self.scan_options}")

        return f"Plan({', '.join(parts)})"

    def __init__(
        self,
        description: str = "",
        start_time: str | int = "2024-02-10T06:00:00",
        duration: float = 60.0,
        sample_rate: float = 20.0,
        frame: str = "ra_dec",
        degrees: bool = True,
        jitter: float = 0,
        scan_center: tuple[float, float] = (4, 10.5),
        scan_pattern: str = "daisy",
        scan_options: dict = {},
    ):

        self.description = description
        self.start_time = start_time
        self.duration = duration
        self.sample_rate = sample_rate
        self.frame = frame
        self.degrees = degrees
        self.jitter = jitter
        self.scan_center = Angle(
            scan_center, units=("degrees" if degrees else "radians")
        )
        self.scan_pattern = scan_pattern
        self.scan_options = scan_options

        if not self.sample_rate > 0:
            raise ValueError("Parameter 'sample_rate' must be greater than zero!")

        # for k, v in PLAN_CONFIGS[self.scan_pattern]["scan_options"].items():
        #     if k not in self.scan_options.keys():
        #         self.scan_options[k] = v

        if not hasattr(self, "start_time"):
            self.start_time = arrow.now().timestamp()
        self.start_time = arrow.get(self.start_time)
        self.end_time = self.start_time.shift(seconds=self.duration)

        self.time_min = self.start_time.timestamp()
        self.time_max = self.end_time.timestamp()
        self.dt = 1 / self.sample_rate

        self.time = np.arange(self.time_min, self.time_max, self.dt)
        self.n_time = len(self.time)

        # # convert radius to width / height
        # if "width" in self.scan_options:
        #     self.scan_options["radius"] = 0.5 * self.scan_options.pop("width")

        # this is in pointing_units
        scan_offsets = get_pattern_generator(self.scan_pattern)(
            self.time,
            **self.scan_options,
        )

        self.scan_offsets = Angle(
            scan_offsets, units=("degrees" if degrees else "radians")
        )

        scan_velocity = np.gradient(
            self.scan_offsets.radians,
            axis=1,
            edge_order=0,
        ) / np.gradient(self.time)

        scan_acceleration = np.gradient(
            scan_velocity,
            axis=1,
            edge_order=0,
        ) / np.gradient(self.time)

        self.max_vel_deg = np.degrees(np.sqrt(np.sum(scan_velocity**2, axis=0)).max())
        self.max_acc_deg = np.degrees(
            np.sqrt(np.sum(scan_acceleration**2, axis=0)).max()
        )

        if self.max_vel_deg > MAX_VELOCITY_WARN:
            logger.warning(
                (
                    f"The maximum velocity of the boresight ({self.max_vel_deg:.01f} deg/s) is "
                    "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
                ),
                stacklevel=2,
            )

        if self.max_acc_deg > MAX_ACCELERATION_WARN:
            logger.warning(
                (
                    f"The maximum acceleration of the boresight ({self.max_acc_deg:.01f} deg/s^2) is "
                    "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
                ),
                stacklevel=2,
            )

        self.scan_offsets.radians += np.radians(
            self.jitter
        ) * np.random.standard_normal(
            size=self.scan_offsets.shape
        )  # noqa

        self.phi, self.theta = coords.dx_dy_to_phi_theta(
            *self.scan_offsets[:],
            *self.scan_center,
        )
        if self.frame == "ra_dec":
            self.ra, self.dec = self.phi, self.theta
        elif self.frame == "az_el":
            self.az, self.el = self.phi, self.theta
        else:
            raise ValueError("Not a valid pointing frame!")

    def plot(self):

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        frame = coords.frames[self.frame]
        label = f"{round(self.scan_center.deg[0], 3)}° {frame['phi_short_name']}, {round(self.scan_center.deg[1], 3)}° {frame['theta_short_name']}"  # noqa

        ax.plot(*self.scan_offsets.values, lw=5e-1)
        ax.scatter(0, 0, c="r", marker="x", label=label)
        ax.set_xlabel(rf"$\Delta \, \theta_x$ [{self.scan_offsets.units_short}]")
        ax.set_ylabel(rf"$\Delta \, \theta_y$ [{self.scan_offsets.units_short}]")
        ax.legend()

    def map_counts(self, instrument=None, x_bins=100, y_bins=100):

        array_offsets = (
            np.zeros((1, 1, 2)) if instrument is None else instrument.offsets[:, None]
        )

        OFFSETS = self.scan_offsets.radians.T[None] + array_offsets

        xmin, ymin = OFFSETS.min(axis=(0, 1))
        xmax, ymax = OFFSETS.max(axis=(0, 1))

        if isinstance(x_bins, int):
            x_bins = np.linspace(xmin, max(xmax, xmin + 1e-6), x_bins + 1)
        if isinstance(y_bins, int):
            y_bins = np.linspace(ymin, max(ymax, ymin + 1e-6), y_bins + 1)

        bs = sp.stats.binned_statistic_2d(
            OFFSETS[..., 1].ravel(),
            OFFSETS[..., 0].ravel(),
            0,
            statistic="count",
            bins=(y_bins, x_bins),
        )

        return x_bins, y_bins, bs[0]

    def plot_counts(self, instrument=None, x_bins=100, y_bins=100):

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        x, y, counts = self.map_counts(
            instrument=instrument, x_bins=x_bins, y_bins=y_bins
        )
        x = Angle(x)
        y = Angle(y)

        heatmap = ax.pcolormesh(x.values, y.values, counts, cmap="turbo", vmin=0)
        ax.set_xlabel(rf"$\Delta \theta_x$ [{x.units_short}]")
        ax.set_ylabel(rf"$\Delta \theta_y$ [{y.units_short}]")

        cbar = fig.colorbar(heatmap, location="right")
        cbar.set_label("counts")
