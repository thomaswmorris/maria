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

from ..coords import Coordinates, Frame, frames, get_center_phi_theta, offsets_to_phi_theta, phi_theta_to_offsets
from ..instrument import Instrument, get_instrument
from ..io import DEFAULT_TIME_FORMAT, read_yaml, repr_lat_lon, repr_phi_theta
from ..site import Site, get_site
from ..units import Quantity
from ..utils import compute_diameter
from .patterns import get_scan_type_generator, scan_types

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

all_patterns = list(scan_types.index.values)

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


def parse_scan_parameters(parameters):

    params = parameters.copy()

    units = "deg" if params.pop("degrees", True) else "rad"

    scan_kwargs = {"offsets": True}
    scan_pattern_kwargs = {"x_throw": 0, "y_throw": 0}

    if "radius" in params:
        scan_pattern_kwargs["x_throw"] = Quantity(params["radius"], units).rad
        scan_pattern_kwargs["y_throw"] = Quantity(params["radius"], units).rad
        params.pop("radius")

    for dim, coord in zip(["x", "y"], ["phi", "theta"]):
        for spec in ["center", "throw"]:
            for frame, entry in frames.T.iterrows():
                key = f"{entry[coord]}_{spec}"
                if key in params:
                    if scan_pattern_kwargs.get("frame", frame) == frame:
                        params["frame"] = frame
                    else:
                        raise ValueError("Multiple frames!")
                    params[f"{dim}_{spec}"] = params.pop(key)

                    if spec == "throw":
                        params["offsets"] = False

    if "frame" not in params:
        logger.warning("Could not infer frame from scan parameters, assuming frame 'ra/dec'")

    params["center"] = np.array(
        [Quantity(params.pop("x_center", 0), units).rad, Quantity(params.pop("y_center", 0), units).rad]
    )

    for param in ["x_throw", "y_throw"]:
        if param in params:
            scan_pattern_kwargs[param] = Quantity(params.pop(param), units).rad

    for param in ["speed"]:
        if param in params:
            scan_pattern_kwargs[param] = Quantity(params.pop(param), f"{units}/s").to("rad/s")

    for param in list(params):
        if param in ["offsets", "frame", "center"]:
            scan_kwargs[param] = params.pop(param)
        else:
            scan_pattern_kwargs[param] = params.pop(param)

    if "frame" not in scan_kwargs:
        logger.warning("Could not infer frame from scan parameters, assuming frame 'ra/dec'")
        scan_kwargs["frame"] = "ra/dec"

    if "speed" not in scan_pattern_kwargs:
        default_speed = min(max(scan_pattern_kwargs["x_throw"], scan_pattern_kwargs["y_throw"]) / 2, np.radians(3))
        scan_pattern_kwargs["speed"] = default_speed if default_speed != 0 else 1

    return scan_kwargs, scan_pattern_kwargs


class Plan:
    @classmethod
    def generate(
        cls,
        instrument: Instrument | str = None,
        site: Site | str = None,
        start_time: str | int = None,
        duration: float = 60.0,
        sample_rate: float = None,
        jitter: float = 0.0,
        roll: float = 0.0,
        scan_type: str = "daisy",
        scan_parameters: dict = {},
        frame: str = None,
        scan_center: tuple[float, float] = None,
    ):

        duration = Quantity(duration, "s")

        if frame is not None and scan_center is not None:
            # warnings.warn("deprecated", DeprecationWarning)

            frame = Frame(frame)

            scan_parameters[f"{frame.phi['name']}_center"] = scan_center[0]
            scan_parameters[f"{frame.theta['name']}_center"] = scan_center[1]

        infer_sample_rate = False
        if sample_rate is None:
            if instrument is None:
                logger.warning("No sample rate passed, assuming 100 Hz")
                sample_rate = 100
            elif isinstance(instrument, str):
                instrument = get_instrument(instrument)
            elif not isinstance(instrument, Instrument):
                raise TypeError()

            sample_rate = 1000
            infer_sample_rate = True

        sample_rate = Quantity(sample_rate, "Hz")

        # for k, v in PLAN_CONFIGS[self.scan_type]["scan_parameters"].items():
        #     if k not in self.scan_parameters.keys():
        #         self.scan_parameters[k] = v

        if start_time is None:
            start_time = arrow.now().timestamp()
        start_time = arrow.get(start_time)

        time_min = start_time.timestamp()
        time_max = time_min + duration.seconds

        time = np.arange(time_min, time_max, 1 / sample_rate.Hz)

        # # convert radius to width / height
        # if "width" in self.scan_parameters:
        #     self.scan_parameters["radius"] = 0.5 * self.scan_parameters.pop("width")

        # this is in pointing_units

        scan_kwargs, scan_pattern_kwargs = parse_scan_parameters(scan_parameters)

        scan_offsets = get_scan_type_generator(scan_type)(
            time,
            **scan_pattern_kwargs,
        )

        logger.debug(f"Generating '{scan_type}' scan with kwargs {scan_kwargs}")
        logger.debug(f"Generating scan with parameters {scan_pattern_kwargs}")

        scan_offsets += np.radians(jitter) * np.random.standard_normal(size=scan_offsets.shape)  # noqa

        assert not np.isnan(scan_offsets).any()

        if scan_kwargs["offsets"]:
            pt = offsets_to_phi_theta(scan_offsets, *scan_kwargs["center"])
        else:
            pt = scan_offsets + scan_kwargs["center"]

        # if len(scan_center) == 2:
        #     units = "deg" if degrees else "rad"
        #     scan_center = (Quantity(scan_center[0], units=units), Quantity(scan_center[1], units=units))
        # else:
        #     raise ValueError("'scan_center' must be a 2-tuple of numbers")

        self = cls(time, phi=pt[..., 0], theta=pt[..., 1], roll=roll, frame=scan_kwargs["frame"], site=site)

        self.generation_kwargs = {"scan_type": scan_type, "scan_parameters": scan_parameters}

        return self

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

    def __init__(
        self,
        time: float,
        phi: float,
        theta: float,
        roll: float = 0.0,
        frame: str = "ra/dec",
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

        if self.frame.name == "ra/dec":
            self.ra = self.phi = phi
            self.dec = self.theta = theta
        elif self.frame.name == "az/el":
            self.az = self.phi = phi
            self.el = self.theta = theta
        elif self.frame.name == "galactic":
            self.glon = self.phi = phi
            self.glat = self.theta = theta
        else:
            raise ValueError("Not a valid pointing frame!")

        offsets = self.offsets()
        self.scan_speed = Quantity(
            np.sqrt(np.square(np.gradient(offsets, axis=0)).sum(axis=1)) / np.gradient(self.time), "rad/s"
        )

        self.roll = roll

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
    def naive(self):
        return self.earth_location is None

    def center(self, frame: str = None, center: tuple[float, float] = None):
        frame = Frame(frame or self.frame.name)
        if center is not None:
            cphi, ctheta = center
        else:
            cphi, ctheta = get_center_phi_theta(
                phi=getattr(self.coords, frame.phi_name), theta=getattr(self.coords, frame.theta_name)
            )
        return (Quantity(cphi, "rad"), Quantity(ctheta, "rad"))

    def offsets(self, frame: str = None, center: tuple[float, float] = None):
        center = center or self.center(frame=frame or self.frame.name)
        phi_theta = np.stack([self.phi, self.theta], axis=-1)
        return phi_theta_to_offsets(phi_theta, float(center[0].rad), float(center[1].rad))

    def plot(self, frames: list[str] | None = None, ax_size: float = 5):

        if frames is None:
            frames = ["az/el", "ra/dec", "galactic"] if self.earth_location is not None else [self.frame.name]

        fig = plt.figure(figsize=(len(frames) * ax_size, ax_size), dpi=256, constrained_layout=True)

        for frame_index, frame in enumerate(frames):
            frame_center = self.coords.center(frame=frame)
            frame_offsets = self.coords.offsets(frame=frame)

            q_frame_offsets = Quantity(frame_offsets, "rad")

            center = self.center()

            xi = q_frame_offsets.human_value[:, 0]
            eta = q_frame_offsets.human_value[:, 1]

            header = fits.header.Header()

            CTYPE1 = Frame(frame).fits["phi"]
            header["CTYPE1"] = f"{CTYPE1}{(5 - len(CTYPE1)) * '-'}SIN" if frame != "az/el" else "GLON-SIN"
            header["CRVAL1"] = frame_center[0].deg
            header["CDELT1"] = -np.degrees(q_frame_offsets.hu["base_units_factor"])
            header["CRPIX1"] = 1
            header["CUNIT1"] = "deg     "

            CTYPE2 = Frame(frame).fits["theta"]
            header["CTYPE2"] = f"{CTYPE2}{(5 - len(CTYPE2)) * '-'}SIN" if frame != "az/el" else "GLAT-SIN"
            header["CRVAL2"] = frame_center[1].deg
            header["CDELT2"] = np.degrees(q_frame_offsets.hu["base_units_factor"])
            header["CRPIX2"] = 1
            header["CUNIT2"] = "deg     "

            wcs = WCS(header)

            ax = fig.add_subplot(1, len(frames), frame_index + 1, projection=wcs)

            # cphi_repr, ctheta_repr = repr_phi_theta(center[0].rad, center[1].rad, frame=self.frame.name, join=True)

            ax.plot(q_frame_offsets.human_value[:, 0], q_frame_offsets.human_value[:, 1], lw=5e-1, c="k")

            # ax.legend(loc="upper right")

            ax.tick_params(axis="x", bottom=True, top=False)
            ax.tick_params(axis="y", left=True, right=False, rotation=90)

            ax2 = ax.secondary_xaxis("top")
            ax2.set_xlabel(rf"$\Delta \, \theta_x$ [${q_frame_offsets.hu['math_name']}$]")

            xmin, ymin = q_frame_offsets.human_value.min(axis=0)
            xmax, ymax = q_frame_offsets.human_value.max(axis=0)
            xcen, ycen = (xmin + xmax) / 2, (ymin + ymax) / 2
            radius = 0.5 * 1.05 * max(ymax - ymin, xmax - xmin)

            if radius == 0:
                radius = Quantity(1, "arcsec").to(q_frame_offsets.human_units)

            ax.scatter(xi[0], eta[0], color="g", marker="+")
            ax.scatter(xi[-1], eta[-1], color="r", marker="+")

            # start_ha = "right" if xi[0] > xi.mean() else "left"
            # start_xytext = (10 if start_ha == "left" else -10, 0)

            # end_ha = "right" if xi[-1] > xi.mean() else "left"
            # end_xytext = (10 if end_ha == "left" else -10, 0)

            annotate_kwargs = dict(
                xycoords="data",
                # textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.3", "fc": "w", "ec": "k", "alpha": 0.75, "lw": 1},
            )

            start_loc, end_loc = list(
                zip(
                    ["left", "right"] if xi[0] < xi[-1] else ["right", "left"],
                    ["lower", "upper"] if eta[0] < eta[-1] else ["upper", "lower"],
                )
            )

            arrowprops = dict(width=1e0, headlength=5e0, headwidth=5e0, shrink=0.05, edgecolor="none")

            ax.annotate(
                xy=(xi[0], eta[0]),
                xytext=(
                    xcen - 0.95 * radius if start_loc[0] == "left" else xcen + 0.95 * radius,
                    ycen - 0.95 * radius if start_loc[1] == "lower" else ycen + 0.95 * radius,
                ),
                text=f"START ({self.repr_start_time})",
                c="green",
                ha="left" if start_loc[0] == "left" else "right",
                va="bottom" if start_loc[1] == "lower" else "top",
                **annotate_kwargs,
                arrowprops={"facecolor": "green", **arrowprops},
            )

            ax.annotate(
                xy=(xi[-1], eta[-1]),
                xytext=(
                    xcen - 0.95 * radius if end_loc[0] == "left" else xcen + 0.95 * radius,
                    ycen - 0.95 * radius if end_loc[1] == "lower" else ycen + 0.95 * radius,
                ),
                text=f"END ({self.repr_end_time})",
                c="red",
                ha="left" if end_loc[0] == "left" else "right",
                va="bottom" if end_loc[1] == "lower" else "top",
                **annotate_kwargs,
                arrowprops={"facecolor": "red", **arrowprops},
            )

            ax.set_xlim(xcen - radius, xcen + radius)
            ax.set_ylim(ycen - radius, ycen + radius)

            ax.grid()
            ax.set_aspect("equal")

            ax.set_xlabel(rf"{Frame(frame).phi_long_name}")
            ax.set_ylabel(rf"{Frame(frame).theta_long_name}")

    def map_counts(self, instrument: Instrument | str = None, x_bins=64, y_bins=64):
        if isinstance(instrument, str):
            instrument = get_instrument(instrument)

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

    def plot_hits(self, instrument=None, x_bins=256, y_bins=256):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

        x, y, counts = self.map_counts(instrument=instrument, x_bins=x_bins, y_bins=y_bins)
        q_x = Quantity(x, "rad")
        q_y = Quantity(y, "rad")

        heatmap = ax.pcolormesh(q_x.human_value, q_y.human_value, counts, cmap="turbo", vmin=0)
        ax.set_xlabel(rf"$\Delta \theta_x$ [${q_x.hu['math_name']}$]")
        ax.set_ylabel(rf"$\Delta \theta_y$ [${q_y.hu['math_name']}$]")

        cbar = fig.colorbar(heatmap, location="right")
        cbar.set_label("counts")

    @property
    def repr_start_time(self):
        return self.start_time.format(DEFAULT_TIME_FORMAT)

    @property
    def repr_end_time(self):
        return self.end_time.format(DEFAULT_TIME_FORMAT)

    def __repr__(self):
        c = self.center(frame=self.frame.name)
        q_offsets = Quantity(self.offsets(), "rad")

        cphi_repr, ctheta_repr = repr_phi_theta(c[0].rad, c[1].rad, frame=self.frame.name, join=True)
        center_string = f"""center:
    {cphi_repr}
    {ctheta_repr}"""

        if not self.naive:
            repr_lat, repr_lon = repr_lat_lon(self.site.latitude.degrees, self.site.longitude.degrees)
            location_string = f"""
    lat: {repr_lat}
    lon: {repr_lon}
    alt: {self.site.altitude}"""
            if self.frame.name != "az/el":
                caz, cel = self.center(frame="az/el")
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
  scan_radius: {Quantity(compute_diameter(q_offsets.rad), "rad")}
  scan_speed(mean): {self.scan_speed.mean()}"""

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Plans can only be added to other Plans")

        assert self.end_time < other.start_time

        time = np.r_[self.time, other.time]
        phi = np.r_[self.phi, other.phi]
        theta = np.r_[self.theta, other.theta]

        return Plan(time=time, phi=phi, theta=theta, frame=self.frame.name, site=self.site)

    def __radd__(self, other):
        return other.__add__(self)
