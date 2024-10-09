import functools
import logging
import time as ttime
from datetime import datetime

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytz
import scipy as sp
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.interpolate import interp1d

from ..utils import repr_lat_lon
from .transforms import (  # noqa
    dx_dy_to_phi_theta,
    get_center_phi_theta,
    phi_theta_to_dx_dy,
    phi_theta_to_xyz,
    xyz_to_phi_theta,
)

logger = logging.getLogger("maria")


frames = {
    "az_el": {
        "astropy_name": "altaz",
        "astropy_phi": "az",
        "astropy_theta": "alt",
        "phi": "az",
        "theta": "el",
        "phi_name": "azimuth",
        "theta_name": "elevation",
    },
    "ra_dec": {
        "astropy_name": "icrs",
        "astropy_phi": "ra",
        "astropy_theta": "dec",
        "phi": "ra",
        "theta": "dec",
        "phi_name": "Right Ascension",
        "theta_name": "Declination",
    },
    "galactic": {
        "astropy_name": "galactic",
        "astropy_phi": "l",
        "astropy_theta": "b",
        "phi": "l",
        "theta": "b",
        "phi_name": "L",
        "theta_name": "B",
    },
}


DEFAULT_EARTH_LOCATION = EarthLocation.from_geodetic(
    0.0, 90.0, height=0.0, ellipsoid=None
)  # noqa
DEFAULT_TIMESTAMP = datetime.now().timestamp()


class Coordinates:
    """
    A class for managing coordinates, allowing us to access different coordinate frames at once.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        r: float = 0.0,
        phi: float = 0.0,
        theta: float = 0.0,
        time: float = DEFAULT_TIMESTAMP,
        earth_location: EarthLocation = DEFAULT_EARTH_LOCATION,
        frame: str = "az_el",
        dtype=np.float64,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.phi = phi
        self.theta = theta
        self.time = time
        self.earth_location = earth_location
        self.frame = frame
        self.dtype = dtype

        # DO NOT BROADCAST TIME. IT STAYS ONE-DIMENSIONAL.
        for attr, value in zip(
            ["x", "y", "z", "r", "phi", "theta", "broadcasted_time"],
            np.broadcast_arrays(x, y, z, r, phi, theta, time)[:-1],
        ):
            if not isinstance(value, dask.array.Array):
                value = da.from_array(value)
            setattr(self, f"_{attr}", value)

        setattr(self, frames[self.frame]["phi"], self._phi)
        setattr(self, frames[self.frame]["theta"], self._theta)

        self.shape = self._phi.shape

        if hasattr(time, "__len__"):
            for axis in range(len(self.time.shape) - 1):
                if (np.ptp(self.time, axis=axis) > 0).any():
                    raise ValueError("Only the last axis can vary in time.")

        ref_time = ttime.monotonic()
        self.compute_transforms()
        duration_ms = 1e3 * (ttime.monotonic() - ref_time)
        logger.debug(
            f"Initialized coordinates with shape {self.shape} in {int(duration_ms)} ms."
        )  # noqa

    def compute_transforms(self):
        self.shaped_time = np.atleast_1d(self.time)
        keep_dims = (-1,) if hasattr(self.time, "__len__") else ()
        time_ordered_center_phi_theta = np.c_[
            get_center_phi_theta(self._phi, self._theta, keep_dims=keep_dims)
        ]

        # (nt) time samples on which to explicitly compute the transformation from astropy
        time_samples_min_res_seconds = 10
        time_samples_min = np.min(self.time) - 1e0
        time_samples_max = np.max(self.time) + 1e0
        n_time_samples = int(
            np.maximum(
                2, (time_samples_max - time_samples_min) / time_samples_min_res_seconds
            )
        )
        self.fid_times = np.linspace(time_samples_min, time_samples_max, n_time_samples)

        sample_indices = interp1d(
            self.shaped_time,
            np.arange(len(self.shaped_time)),
            bounds_error=False,
            kind="nearest",
            fill_value="extrapolate",
        )(self.fid_times).astype(int)

        # three fiducial offsets from the boresight to train a transformation matrix
        # shape: (n_fid, 2)
        fid_offsets = np.radians([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        psi = np.linspace(0, 2 * np.pi, 12 + 1)[:-1]
        fid_offsets = np.concatenate(
            [r * np.c_[np.cos(psi), np.sin(psi)] for r in [1, 10, 30]], axis=0
        )

        self.fid_phi, self.fid_theta = dx_dy_to_phi_theta(
            fid_offsets[..., 0],
            fid_offsets[..., 1],
            time_ordered_center_phi_theta[:, 0][sample_indices][..., None],
            time_ordered_center_phi_theta[:, 1][sample_indices][..., None],
        )

        self.fid_skycoords = {
            self.frame: SkyCoord(
                self.fid_phi * u.rad,
                self.fid_theta * u.rad,
                obstime=Time(
                    self.fid_times[:, None],
                    format="unix",
                ),
                frame=frames[self.frame]["astropy_name"],
                location=self.earth_location,
            )
        }

        self.transforms = {}
        self.fid_points = {self.frame: phi_theta_to_xyz(self.fid_phi, self.fid_theta)}
        A = self.fid_points[self.frame]
        AT = np.swapaxes(self.fid_points[self.frame], -2, -1)

        for frame, config in frames.items():
            if frame in self.fid_skycoords:
                continue

            self.fid_skycoords[frame] = getattr(
                self.fid_skycoords[self.frame], config["astropy_name"]
            )

            frame_fid_phi = getattr(
                self.fid_skycoords[frame], frames[frame]["astropy_phi"]
            ).rad
            frame_fid_theta = getattr(
                self.fid_skycoords[frame], frames[frame]["astropy_theta"]
            ).rad

            self.fid_points[frame] = phi_theta_to_xyz(frame_fid_phi, frame_fid_theta)

            # voodoo!
            self.transforms[frame] = (
                np.linalg.inv(AT @ A)
                @ AT
                @ phi_theta_to_xyz(frame_fid_phi, frame_fid_theta)
            )

            transform_stack = interp1d(
                self.fid_times,
                self.transforms[frame],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                axis=0,
            )(self.time)

            frame_phi, frame_theta = xyz_to_phi_theta(
                (np.expand_dims(self.compute_points(), -2) @ transform_stack).squeeze()
            )

            setattr(self, frames[frame]["phi"], frame_phi)
            setattr(self, frames[frame]["theta"], frame_theta)

    @property
    def timestep(self):
        if len(self.time):
            return np.mean(np.gradient(self.time))
        return None

    def downsample(self, timestep: float = None, factor: int = None):
        if timestep is None and factor is None:
            raise ValueError("You must supply either 'timestep' or 'factor'.")

        timestep = timestep or factor * self.timestep

        ds_time = np.arange(self.time.min(), self.time.max(), timestep)
        ds_phi = sp.interpolate.interp1d(self.time, self._phi, axis=-1)(ds_time)
        ds_theta = sp.interpolate.interp1d(self.time, self._theta, axis=-1)(ds_time)

        return Coordinates(
            time=ds_time,
            phi=ds_phi,
            theta=ds_theta,
            earth_location=self.earth_location,
            frame=self.frame,
            dtype=self.dtype,
        )

    def boresight(self):
        cphi, ctheta = get_center_phi_theta(self._phi, self.theta, keep_dims=(-1,))

        return Coordinates(
            time=self.time,
            phi=cphi,
            theta=ctheta,
            earth_location=self.earth_location,
            frame=self.frame,
            dtype=self.dtype,
        )

    @functools.cached_property
    def summary(self):
        # compute summary for the string repr
        summary = pd.DataFrame(columns=["min", "mean", "max"])
        boresight = self.boresight
        for attr in ["az", "el", "ra", "dec"]:
            for stat in ["min", "mean", "max"]:
                summary.loc[
                    attr, stat
                ] = f"{float(np.degrees(getattr(getattr(boresight, attr), stat)().compute())):.03f}°"

        return summary

    def project(self, z, frame="az_el"):
        # if not ((h is None) ^ (z is None)):
        #     raise ValueError("You must specify exactly one of 'h' or 'z'.")

        phi = getattr(self, frames[frame]["phi"])
        theta = getattr(self, frames[frame]["theta"])

        tan_theta = np.tan(theta)[..., None]
        p = (z - self.z) * np.concatenate(
            [
                np.cos(phi)[..., None] / tan_theta,
                np.sin(phi)[..., None] / tan_theta,
                np.ones((*phi.shape, 1)),
            ],
            axis=-1,
        )

        return p + np.c_[self.x, self.y, self.z][None]

    def compute_points(self):
        return phi_theta_to_xyz(self._phi, self._theta)

    def center(self, frame=None):
        frame = frame or self.frame
        return get_center_phi_theta(
            getattr(self, frames[frame]["phi"]), getattr(self, frames[frame]["theta"])
        )

    def broadcast(self, offsets, frame, axis=0):
        phi, theta = dx_dy_to_phi_theta(*offsets.T[..., None], self.az, self.el)
        return Coordinates(
            time=self.time,
            phi=phi.compute(),
            theta=theta.compute(),
            earth_location=self.earth_location,
            frame=frame,
        )

    def __getitem__(self, i):
        """
        TODO: error handling for slicing
        """
        if not isinstance(i, tuple):
            i = tuple(
                [
                    i,
                ]
            )  # noqa
        *_, time_slice = i
        return Coordinates(
            time=self.time[time_slice],
            phi=self._phi[i],
            theta=self._theta[i],
            earth_location=self.earth_location,
            frame=self.frame,
        )

    def offsets(self, frame, center="auto", units="radians"):
        if isinstance(center, str):
            if center == "auto":
                center = self.center(frame=frame)
        if frame == "az_el":
            dx, dy = phi_theta_to_dx_dy(self.az, self.el, *center)
        elif frame == "ra_dec":
            dx, dy = phi_theta_to_dx_dy(self.ra, self.dec, *center)
        elif frame == "galactic":
            dx, dy = phi_theta_to_dx_dy(self.l, self.b, *center)

        if units == "radians":
            return dx, dy
        elif units == "degrees":
            return np.degrees(dx), np.degrees(dy)
        elif units == "arcmin":
            return 60 * np.degrees(dx), 60 * np.degrees(dy)
        elif units == "arcsec":
            return 3600 * np.degrees(dx), 3600 * np.degrees(dy)

    def __repr__(self):
        lon = self.earth_location.lon.deg
        lat = self.earth_location.lat.deg

        date_string = (
            datetime.fromtimestamp(np.mean(self.time))
            .astimezone(pytz.utc)
            .strftime("%Y %h %-d %H:%M:%S")
        )

        return f"Coordinates(shape={self.shape}, earth_location=({repr_lat_lon(lat, lon)}), time='{date_string} UTC')"
