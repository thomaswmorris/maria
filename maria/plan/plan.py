from __future__ import annotations

import logging
import os
from pathlib import Path

import arrow
import numpy as np
import pandas as pd
import scipy as sp
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import pyplot as plt

from .. import coords
from ..coords import Coordinates, frames, get_center_phi_theta, phi_theta_to_offsets
from ..io import DEFAULT_TIME_FORMAT, repr_lat_lon, repr_phi_theta
from ..site import Site, get_site
from ..units import Quantity
from ..utils import compute_diameter, read_yaml
from .patterns import get_scan_pattern_generator, scan_patterns

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


class Plan:
    """
    A dataclass containing time-ordered plan data.
    """

    @classmethod
    def generate(
        cls,
        site: Site | str = None,
        description: str = "",
        start_time: str | int = None,
        duration: float = 60.0,
        sample_rate: float = 50.0,
        frame: str = "ra_dec",
        degrees: bool = True,
        jitter: float = 0,
        scan_center: tuple[float, float] = (0.0, 0.0),
        scan_pattern: str = "daisy",
        scan_options: dict = {},
    ):
        duration = Quantity(duration, "s")
        sample_rate = Quantity(sample_rate, "Hz")

        # for k, v in PLAN_CONFIGS[self.scan_pattern]["scan_options"].items():
        #     if k not in self.scan_options.keys():
        #         self.scan_options[k] = v

        if start_time is None:
            start_time = arrow.now().timestamp()
        start_time = arrow.get(start_time)

        time_min = start_time.timestamp()
        time_max = time_min + duration.seconds

        time = np.arange(time_min, time_max, 1 / sample_rate.Hz)

        # # convert radius to width / height
        # if "width" in self.scan_options:
        #     self.scan_options["radius"] = 0.5 * self.scan_options.pop("width")

        # this is in pointing_units
        scan_offsets = get_scan_pattern_generator(scan_pattern)(
            time,
            **scan_options,
        )

        scan_offsets = Quantity(scan_offsets, units=("deg" if degrees else "rad")).rad

        # TODO: scan speed checks in az/el frame

        # scan_velocity = np.gradient(
        #     self.scan_offsets,
        #     axis=1,
        #     edge_order=0,
        # ) / np.gradient(self.time)

        # scan_acceleration = np.gradient(
        #     scan_velocity,
        #     axis=1,
        #     edge_order=0,
        # ) / np.gradient(self.time)

        # self.max_vel_deg = np.degrees(np.sqrt(np.sum(scan_velocity**2, axis=0)).max())
        # self.max_acc_deg = np.degrees(np.sqrt(np.sum(scan_acceleration**2, axis=0)).max())

        # if self.max_vel_deg > MAX_VELOCITY_WARN:
        #     logger.warning(
        #         (
        #             f"The maximum velocity of the boresight ({self.max_vel_deg:.01f} deg/s) is "
        #             "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
        #         ),
        #         stacklevel=2,
        #     )

        # if self.max_acc_deg > MAX_ACCELERATION_WARN:
        #     logger.warning(
        #         (
        #             f"The maximum acceleration of the boresight ({self.max_acc_deg:.01f} deg/s^2) is "
        #             "physically unrealistic. If this is undesired, double-check the parameters for your scan strategy."
        #         ),
        #         stacklevel=2,
        #     )

        if len(scan_center) == 2:
            units = "deg" if degrees else "rad"
            scan_center = (Quantity(scan_center[0], units=units), Quantity(scan_center[1], units=units))
        else:
            raise ValueError("'scan_center' must be a 2-tuple of numbers")

        scan_offsets += np.radians(jitter) * np.random.standard_normal(size=scan_offsets.shape)  # noqa

        pt = coords.offsets_to_phi_theta(
            scan_offsets.T,
            scan_center[0].rad,
            scan_center[1].rad,
        )

        self = cls(time, phi=pt[..., 0], theta=pt[..., 1], frame=frame, site=site)

        self.generation_kwargs = {"scan_pattern": scan_pattern, "scan_options": scan_options}

        return self

    def __init__(
        self,
        time: float,
        phi: float,
        theta: float,
        frame: str = "ra_dec",
        site: Site | str = None,
        latitude: float = None,  # in degrees
        longitude: float = None,  # in degrees
        altitude: float = 0,  # in meters
    ):
        if site is not None:
            if isinstance(site, str):
                site = get_site(site)
            elif not isinstance(site, Site):
                raise TypeError()

            self.site = site
            earth_location = site.earth_location

        elif latitude is not None and longitude is not None:
            # self.latitude = Quantity(latitude, "deg")
            # self.longitude = Quantity(longitude, "deg")
            # self.altitude = Quantity(altitude, "m")
            earth_location = EarthLocation.from_geodetic(
                lon=longitude,
                lat=latitude,
                height=altitude,
            )

        else:
            earth_location = None

        self.coords = Coordinates(
            phi=phi,
            theta=theta,
            t=time,
            frame=frame,
            earth_location=earth_location,
        )

        if self.frame == "ra_dec":
            self.ra = self.phi = phi
            self.dec = self.theta = theta
        elif self.frame == "az_el":
            self.az = self.phi = phi
            self.el = self.theta = theta
        else:
            raise ValueError("Not a valid pointing frame!")

    @property
    def n(self):
        return len(self.coords.t)

    @property
    def time(self):
        return self.coords.t

    @property
    def frame(self):
        return self.coords.frame

    @property
    def earth_location(self):
        return self.coords.earth_location

    @property
    def dt(self):
        return np.median(np.diff(self.coords.t))

    @property
    def sample_rate(self):
        return Quantity(1 / self.dt, "Hz")

    @property
    def duration(self):
        return Quantity(np.ptp(self.coords.t) + self.dt, "s")

    @property
    def start_time(self):
        return arrow.get(self.time[0])

    @property
    def end_time(self):
        return self.start_time.shift(seconds=self.duration.s)

    @property
    def frame_data(self):
        return frames[self.frame]

    @property
    def naive(self):
        return self.earth_location is None

    def center(self, frame: str = None):
        frame = frame or self.frame
        frame_data = frames[frame]
        cphi, ctheta = get_center_phi_theta(
            phi=getattr(self.coords, frame_data["phi"]), theta=getattr(self.coords, frame_data["theta"])
        )
        return (Quantity(cphi, "rad"), Quantity(ctheta, "rad"))

    def offsets(self, frame: str = None):
        frame = frame or self.frame
        center = self.center(frame=frame)
        phi_theta = np.stack([self.phi, self.theta], axis=-1)
        return phi_theta_to_offsets(phi_theta, float(center[0].rad), float(center[1].rad))

    def plot(self, plot_az_el: bool = None):
        two_panel = plot_az_el if plot_az_el is not None else (self.frame != "az_el" and not self.naive)

        q_offsets = Quantity(self.offsets(), "rad")
        center = self.center()

        header = fits.header.Header()
        header["CDELT1"] = -q_offsets.u["factor"]
        header["CDELT2"] = q_offsets.u["factor"]
        header["CRPIX1"] = 1
        header["CRPIX2"] = 1
        header["CTYPE1"] = "RA---SIN"
        header["CUNIT1"] = "deg     "
        header["CTYPE2"] = "DEC--SIN"
        header["CUNIT2"] = "deg     "
        header["RADESYS"] = "FK5     "
        header["CRVAL1"] = center[0].deg
        header["CRVAL2"] = center[1].deg

        wcs = WCS(header)

        fig = plt.figure(figsize=(5, 5) if self.naive else (10, 5), dpi=256, constrained_layout=True)
        ax = fig.add_subplot(111 if two_panel is None else 121, projection=wcs)

        cphi_repr, ctheta_repr = repr_phi_theta(center[0].rad, center[1].rad, frame=self.frame)

        ax.plot(q_offsets.value[:, 0], q_offsets.value[:, 1], lw=5e-1)
        ax.scatter(0, 0, c="r", marker="x", label=f"{cphi_repr}\n{ctheta_repr}")

        ax.legend(loc="upper right")

        ax.set_aspect("equal")

        ax.tick_params(axis="x", bottom=True, top=False)
        ax.tick_params(axis="y", left=True, right=False, rotation=90)

        ax2 = ax.secondary_xaxis("top")
        ax2.set_xlabel(rf"$\Delta \, \theta_x$ [${q_offsets.u['math_name']}$]")
        ax.set_xlabel(rf"{self.frame_data['phi_long_name']}")
        ax.set_ylabel(rf"{self.frame_data['theta_long_name']}")

        if not two_panel:
            ay2 = ax.secondary_yaxis("right")
            ay2.set_ylabel(rf"$\Delta \, \theta_y$ [${q_offsets.u['math_name']}$]")

        if two_panel:
            az = Quantity(np.unwrap(self.coords.az), "rad")
            el = Quantity(self.coords.el, "rad")

            ax = fig.add_subplot(122)

            start_ha = "right" if az[0] > az.mean() else "left"
            start_xytext = (10 if start_ha == "left" else -10, 0)

            end_ha = "right" if az[-1] > az.mean() else "left"
            end_xytext = (10 if end_ha == "left" else -10, 0)

            annotate_kwargs = dict(
                va="center",
                xycoords="data",
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.3", "fc": "w", "ec": "k", "alpha": 0.5, "lw": 1},
            )

            ax.scatter(az[0].deg, el[0].deg, color="g", marker="+")
            ax.scatter(az[-1].deg, el[-1].deg, color="r", marker="+")
            ax.annotate(
                xy=(az[0].deg, el[0].deg),
                xytext=start_xytext,
                text=f"START ({self.repr_start_time})",
                c="g",
                ha=start_ha,
                **annotate_kwargs,
            )
            ax.annotate(
                xy=(az[-1].deg, el[-1].deg),
                xytext=end_xytext,
                text=f"END ({self.repr_end_time})",
                c="r",
                ha=end_ha,
                **annotate_kwargs,
            )

            ax.plot(az.deg, el.deg, lw=5e-1)

            ax.set_xlabel("Azimuth [degrees]")
            ax.set_ylabel("Elevation [degrees]")

            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    def map_counts(self, instrument=None, x_bins=64, y_bins=64):
        array_offsets = np.zeros((1, 1, 2)) if instrument is None else instrument.offsets[:, None]

        OFFSETS = self.offsets()[None] + array_offsets

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

    def plot_counts(self, instrument=None, x_bins=64, y_bins=64):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        x, y, counts = self.map_counts(instrument=instrument, x_bins=x_bins, y_bins=y_bins)
        q_x = Quantity(x, "rad")
        q_y = Quantity(y, "rad")

        heatmap = ax.pcolormesh(q_x.value, q_y.value, counts, cmap="turbo", vmin=0)
        ax.set_xlabel(rf"$\Delta \theta_x$ [{q_x.u['math_name']}]")
        ax.set_ylabel(rf"$\Delta \theta_y$ [{q_y.u['math_name']}]")

        cbar = fig.colorbar(heatmap, location="right")
        cbar.set_label("counts")

    @property
    def repr_start_time(self):
        return self.start_time.format(DEFAULT_TIME_FORMAT)

    @property
    def repr_end_time(self):
        return self.end_time.format(DEFAULT_TIME_FORMAT)

    def __repr__(self):
        c = self.center(frame=self.frame)
        q_offsets = Quantity(self.offsets(), "rad")

        cphi_repr, ctheta_repr = repr_phi_theta(c[0].rad, c[1].rad, frame=self.frame)
        center_string = f"""center:
    {cphi_repr}
    {ctheta_repr}"""

        if not self.naive:
            repr_lat, repr_lon = repr_lat_lon(self.site.latitude.degrees, self.site.longitude.degrees)
            location_string = f"""
    lat: {repr_lat}
    lon: {repr_lon}
    alt: {self.site.altitude}"""
            if self.frame != "az_el":
                caz, cel = self.center(frame="az_el")
                center_string = f"""center:
    {cphi_repr}
    {ctheta_repr}
    az(mean): {caz}
    el(mean): {cel}"""
        else:
            location_string = f"naive"

        return f"""Plan:
  duration: {self.duration}
    start: {self.repr_start_time}
    end:   {self.repr_end_time}
  location: {location_string}
  sample_rate: {self.sample_rate}
  {center_string}
  scan_radius: {Quantity(compute_diameter(q_offsets.rad), "rad")}"""

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Plans can only be added to other Plans")

        assert self.end_time < other.start_time

        time = np.r_[self.time, other.time]
        phi = np.r_[self.phi, other.phi]
        theta = np.r_[self.theta, other.theta]

        return Plan(time=time, phi=phi, theta=theta, frame=self.frame, site=self.site)

    def __radd__(self, other):
        return other.__add__(self)
