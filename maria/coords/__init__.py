import functools
import time as ttime
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from scipy.interpolate import interp1d


def now():
    return ttime.time()


def dx_dy_to_phi_theta(dx, dy, cphi, ctheta):
    """
    A fast and well-conditioned to convert from local dx/dy coordinates to phi/theta coordinates.
    """

    if not dx.shape == dy.shape:
        raise ValueError(
            f"The shapes of 'dx' and 'dy' must be the same. Got shapes {np.shape(dx)} and {np.shape(dy)}"
        )

    r = np.sqrt(dx**2 + dy**2)  # distance from the center
    p = np.arctan2(dx, -dy)  # 0 at the bottom, increases CCW to pi at the top

    # if we're looking at the north pole, we have (lon, lat) = (p, pi/2 - r)
    # a projection looking from the east
    proj_from_east = (np.sin(r) * np.cos(p) + 1j * np.cos(r)) * np.exp(
        1j * (ctheta - np.pi / 2)
    )
    phi = cphi + np.arctan2(np.sin(r) * np.sin(p), np.real(proj_from_east))
    theta = np.arcsin(np.imag(proj_from_east))

    return (
        phi,
        theta,
    )


def phi_theta_to_dx_dy(phi, theta, cphi, ctheta):
    """
    A fast and well-conditioned to convert from phi/theta coordinates to local dx/dy coordinates.
    """

    if not phi.shape == theta.shape:
        raise ValueError(
            f"The shapes of 'phi' and 'theta' must be the same. Got shapes {np.shape(phi)} and {np.shape(theta)}"
        )

    dphi = phi - cphi
    proj_from_east = (np.cos(dphi) * np.cos(theta) + 1j * np.sin(theta)) * np.exp(
        1j * (np.pi / 2 - ctheta)
    )
    dz = np.sin(dphi) * np.cos(theta) + 1j * np.real(proj_from_east)
    r = np.abs(dz)
    dz *= np.arcsin(r) / r

    # negative, because we're looking at the observer
    return np.real(dz), -np.imag(dz)


def phi_theta_to_xyz(phi, theta):
    """
    Project a longitude and lattitude onto the unit sphere.
    """
    # you can add a newaxis on numpy floats, but not python floats. who knew?
    return np.concatenate(
        [
            (np.cos(phi) * np.cos(theta))[..., None],
            (np.sin(phi) * np.cos(theta))[..., None],
            (np.sin(theta))[..., None],
        ],
        axis=-1,
    )


def xyz_to_phi_theta(xyz):
    """
    Find the longitude and latitude of a 3-vector.
    """
    return np.arctan2(xyz[..., 1], xyz[..., 0]) % (2 * np.pi), np.arcsin(
        xyz[..., 2] / np.sqrt(np.sum(xyz**2, axis=-1))
    )


def get_center_phi_theta(phi, theta, keep_last_dim=False):
    """ """
    xyz = phi_theta_to_xyz(phi, theta)

    if keep_last_dim:
        center_xyz = xyz.mean(axis=tuple(range(xyz.ndim - 2)))
        center_xyz /= np.sqrt(np.sum(np.square(center_xyz), axis=-1))[..., None]
    else:
        center_xyz = xyz.mean(axis=tuple(range(xyz.ndim - 1)))
        center_xyz /= np.sqrt(np.sum(np.square(center_xyz)))

    return xyz_to_phi_theta(center_xyz)


frames = {
    "az_el": {
        "astropy_name": "altaz",
        "astropy_phi": "az",
        "astropy_theta": "alt",
        "phi": "az",
        "theta": "el",
        "phi_name": "RA",
        "theta_name": "Dec.",
    },
    "ra_dec": {
        "astropy_name": "icrs",
        "astropy_phi": "ra",
        "astropy_theta": "dec",
        "phi": "ra",
        "theta": "dec",
        "phi_name": "RA",
        "theta_name": "Dec.",
    },
    "galactic": {
        "astropy_name": "galactic",
        "astropy_phi": "l",
        "astropy_theta": "b",
        "phi": "l",
        "theta": "b",
        "phi_name": "RA",
        "theta_name": "Dec.",
    },
}


class Angle:
    def __init__(self, a, units="radians"):
        if units == "radians":
            self.a = a
        elif units == "degrees":
            self.a = (np.pi / 180) * a
        elif units == "arcmin":
            self.a = (np.pi / 180 / 60) * a
        elif units == "arcsec":
            self.a = (np.pi / 180 / 3600) * a
        else:
            raise ValueError(
                "'units' must be one of ['radians', 'degrees', 'arcmin', 'arcsec']"
            )

        self.is_scalar = len(np.shape(self.a)) == 0

        if not self.is_scalar:
            self.a = np.unwrap(self.a)

    @property
    def radians(self):
        return self.a

    @property
    def rad(self):
        return self.radians

    @property
    def degrees(self):
        return (180 / np.pi) * self.a

    @property
    def deg(self):
        return self.degrees

    @property
    def arcmin(self):
        return (60 * 180 / np.pi) * self.a

    @property
    def arcsec(self):
        return (3600 * 180 / np.pi) * self.a

    def __repr__(self):
        return f"Angle(units='radians', value={self.a.__repr__()})"

    @property
    def units(self):
        # peak-to-peak
        max_deg = self.deg if self.is_scalar else self.deg.max()

        if max_deg < 0.5 / 60:
            return "arcsec"
        if max_deg < 0.5:
            return "arcmin"

        return "degrees"


class Coordinates:
    """
    A class for managing coordinates, allowing us to access different coordinate frames at once.
    """

    def __init__(
        self,
        time: float,
        phi: float,
        theta: float,
        location: EarthLocation,
        frame: str = "ra_dec",
        dtype=np.float32,
        time_offset: float = 0,
    ):
        if float(time.min()) != 0:
            time_offset = time.min()
            time -= time_offset

        self.phi = np.atleast_1d(
            (phi if isinstance(phi, da.Array) else da.from_array(phi))
        ).astype(dtype)
        self.theta = np.atleast_1d(
            (theta if isinstance(theta, da.Array) else da.from_array(theta))
        ).astype(dtype)
        self.time = np.atleast_1d(
            (time if isinstance(time, da.Array) else da.from_array(time))
        ).astype(dtype)

        self.timestep = np.median(np.gradient(self.time))

        if self.time.ndim > 1:
            raise ValueError("'time' can be at most one-dimensional.")
        if (
            len(self.time) != self.phi.shape[-1]
            or len(self.time) != self.theta.shape[-1]
        ):
            raise ValueError(
                "The size of the last dimensions of [phi, theta, time] must all match."
            )

        *self.input_shape, self.time_shape = self.phi.shape

        self.location = location
        self.frame = frame
        self.time_offset = time_offset
        self.dtype = dtype

        _center_phi, _center_theta = get_center_phi_theta(
            self.phi, self.theta, keep_last_dim=True
        )

        # three fiducial offsets from the boresight to train a transformation matrix
        fid_offsets = np.radians([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # the time-ordered coordinates of those offsets in the frame
        time_ordered_fid_phi, time_ordered_fid_theta = dx_dy_to_phi_theta(
            *fid_offsets.T, _center_phi[:, None], _center_theta[:, None]
        )

        time_ordered_fid_points = phi_theta_to_xyz(
            time_ordered_fid_phi, time_ordered_fid_theta
        )

        # time_offset = _time.min().compute()
        # end_time = _time.max().compute()
        duration = self.time.max().compute()

        self.sampled_time = np.linspace(0, duration, int(np.maximum(2, duration / 1)))

        self.sampled_fid_points = interp1d(self.time, time_ordered_fid_points, axis=0)(
            self.sampled_time
        )
        self.sampled_fid_points_inv = np.linalg.inv(self.sampled_fid_points)
        sampled_fid_phi, sampled_fid_theta = xyz_to_phi_theta(self.sampled_fid_points)

        # call astropy to compute a literal representation of the sampled fiducial points
        self.sampled_fid_skycoords = SkyCoord(
            sampled_fid_phi * u.rad,
            sampled_fid_theta * u.rad,
            obstime=Time(
                self.time_offset + np.kron(np.ones((3, 1)), self.sampled_time).T,
                format="unix",
            ),
            frame=frames[frame]["astropy_name"],
            location=location,
        )

        self.TRANSFORMS = {}

    def downsample(self, timestep: float = None, factor: int = None):
        if timestep is None and factor is None:
            raise ValueError("You must supply either 'timestep' or 'factor'.")

        timestep = timestep or factor * self.timestep

        ds_time = np.arange(self.time.min(), self.time.max(), timestep)
        ds_phi = sp.interpolate.interp1d(self.time, self.phi, axis=-1)(ds_time)
        ds_theta = sp.interpolate.interp1d(self.time, self.theta, axis=-1)(ds_time)

        return Coordinates(
            time=ds_time,
            phi=ds_phi,
            theta=ds_theta,
            location=self.location,
            frame=self.frame,
            dtype=self.dtype,
            time_offset=self.time_offset,
        )

    @functools.cached_property
    def boresight(self):
        cphi, ctheta = get_center_phi_theta(self.phi, self.theta, keep_last_dim=True)

        return Coordinates(
            time=self.time,
            phi=cphi,
            theta=ctheta,
            location=self.location,
            frame=self.frame,
            dtype=self.dtype,
            time_offset=self.time_offset,
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

    def compute_points(self):
        return phi_theta_to_xyz(self.phi, self.theta)

    def to_frame(self, frame):
        """
        Convert to a frame. This is expensive to compute, so we cache the output.
        """

        if frame == self.frame:
            return self.phi, self.theta

        elif frame == "az_el":
            sampled_fid_skycoords_new_frame = self.sampled_fid_skycoords.altaz

        elif frame == "ra_dec":
            sampled_fid_skycoords_new_frame = self.sampled_fid_skycoords.icrs

        elif frame == "galactic":
            sampled_fid_skycoords_new_frame = self.sampled_fid_skycoords.galactic

        else:
            raise ValueError("'frame' must be one of ['az_el', 'ra_dec', 'galactic']")

        # find a rotation matrix between the init frame and the new frame
        sampled_fid_phi_new_frame = getattr(
            sampled_fid_skycoords_new_frame, frames[frame]["astropy_phi"]
        ).rad
        sampled_fid_theta_new_frame = getattr(
            sampled_fid_skycoords_new_frame, frames[frame]["astropy_theta"]
        ).rad
        sampled_fid_points_new_frame = phi_theta_to_xyz(
            sampled_fid_phi_new_frame, sampled_fid_theta_new_frame
        )
        sampled_rotation_matrix = (
            self.sampled_fid_points_inv @ sampled_fid_points_new_frame
        ).swapaxes(-2, -1)

        # apply the matrix and convert to coordinates
        extra_dims = tuple(range(len(self.input_shape)))
        transform = np.expand_dims(
            interp1d(self.sampled_time, sampled_rotation_matrix, axis=0)(self.time),
            axis=extra_dims,
        )
        return xyz_to_phi_theta(
            (
                transform.astype(self.dtype)
                @ np.expand_dims(self.compute_points(), axis=-1)
            ).squeeze()
        )

    @functools.cached_property
    def az_el(self):
        return self.to_frame(frame="az_el")

    @functools.cached_property
    def center_az_el(self):
        return get_center_phi_theta(*self.az_el)

    @functools.cached_property
    def ra_dec(self):
        return self.to_frame(frame="ra_dec")

    @functools.cached_property
    def center_ra_dec(self):
        return get_center_phi_theta(*self.ra_dec)

    @functools.cached_property
    def galactic(self):
        return self.to_frame(frame="galactic")

    @functools.cached_property
    def center_galactic(self):
        return get_center_phi_theta(*self.galactic)

    def __getattr__(self, attr: str) -> Any:
        if attr == "az":
            return self.az_el[0]
        if attr == "el":
            return self.az_el[1]
        if attr == "center_az":
            return self.dtype(self.center_az_el[0])
        if attr == "center_el":
            return self.dtype(self.center_az_el[1])

        if attr == "ra":
            return self.ra_dec[0]
        if attr == "dec":
            return self.ra_dec[1]
        if attr == "center_ra":
            return self.dtype(self.center_ra_dec[0])
        if attr == "center_dec":
            return self.dtype(self.center_ra_dec[1])

        if attr == "l":
            return self.galactic[0]
        if attr == "b":
            return self.galactic[1]
        if attr == "center_l":
            return self.dtype(self.center_galactic[0])
        if attr == "center_b":
            return self.dtype(self.center_galactic[1])

        raise AttributeError(f"Coordinates object has no attribute named '{attr}'.")

    def offsets(self, frame, center="auto", units="radians"):
        if isinstance(center, str):
            if center == "auto":
                center = get_center_phi_theta(*self.to_frame(frame))

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
        return self.summary.__repr__()

    def _repr_html_(self):
        return self.summary._repr_html_()
