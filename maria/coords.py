import numpy as np
import scipy as sp
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .utils.coords import dx_dy_to_phi_theta, phi_theta_to_dx_dy


def phi_theta_to_xyz(phi, theta):
    """ """
    x = np.cos(phi) * np.cos(theta)
    y = np.sin(phi) * np.cos(theta)
    z = np.sin(theta)

    # you can add a newaxis on numpy floats, but not python floats. who knew?
    xyz = np.r_[x[None], y[None], z[None]]

    return np.moveaxis(xyz, 0, -1)


def xyz_to_phi_theta(xyz):
    """ """
    return np.arctan2(xyz[..., 1], xyz[..., 0]) % (2 * np.pi), np.arcsin(xyz[..., 2])


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


FRAMES = {
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


class Coordinates:
    """
    A class for managing coordinates, allowing us to access different coordinate frames at once.
    """

    def __init__(
        self, time: float, phi: float, theta: float, location: EarthLocation, frame: str
    ):
        self.time = time
        self.phi = phi
        self.theta = theta
        self.location = location
        self.frame = frame

        _phi = np.atleast_2d(phi)
        _theta = np.atleast_2d(theta)
        _time = np.atleast_1d(time)

        _points = phi_theta_to_xyz(_phi, _theta)
        _center_points = _points.mean(axis=tuple(range(_points.ndim - 2)))
        _center_points /= np.sqrt(np.sum(np.square(_center_points), axis=-1))[..., None]
        _center_phi, _center_theta = xyz_to_phi_theta(_center_points)

        fid_dx = np.radians([-1e0, +1e0, +0e0])
        fid_dy = np.radians([+0e0, +0e0, +1e0])

        FID_PHI, FID_THETA = dx_dy_to_phi_theta(
            fid_dx, fid_dy, _center_phi[:, None], _center_theta[:, None]
        )

        downsample_time = np.linspace(
            _time.min(), _time.max(), int(np.maximum(2, _time.ptp() / 1))
        )

        DS_FID_PHI = sp.interpolate.interp1d(_time, FID_PHI, axis=0)(downsample_time)
        DS_FID_THETA = sp.interpolate.interp1d(_time, FID_THETA, axis=0)(
            downsample_time
        )
        DS_FID_TIME = np.ones(3) * downsample_time[:, None]
        DS_FID_POINTS = np.swapaxes(phi_theta_to_xyz(DS_FID_PHI, DS_FID_THETA), -2, -1)

        DS_FID_COORDS = SkyCoord(
            DS_FID_PHI * u.rad,
            DS_FID_THETA * u.rad,
            obstime=Time(DS_FID_TIME, format="unix"),
            frame=FRAMES[frame]["astropy_name"],
            location=location,
        )

        self.TRANSFORMS = {}

        for new_frame in ["ra_dec", "az_el", "galactic"]:
            if new_frame == self.frame:
                DS_FID_COORDS_NEW_FRAME = DS_FID_COORDS

            elif new_frame == "az_el":
                DS_FID_COORDS_NEW_FRAME = DS_FID_COORDS.altaz

            elif new_frame == "ra_dec":
                DS_FID_COORDS_NEW_FRAME = DS_FID_COORDS.icrs

            elif new_frame == "galactic":
                DS_FID_COORDS_NEW_FRAME = DS_FID_COORDS.galactic

            DS_FID_PHI_NEW_FRAME = getattr(
                DS_FID_COORDS_NEW_FRAME, FRAMES[new_frame]["astropy_phi"]
            ).rad
            DS_FID_THETA_NEW_FRAME = getattr(
                DS_FID_COORDS_NEW_FRAME, FRAMES[new_frame]["astropy_theta"]
            ).rad

            DS_FID_POINTS_NEW_FRAME = np.swapaxes(
                phi_theta_to_xyz(DS_FID_PHI_NEW_FRAME, DS_FID_THETA_NEW_FRAME), -2, -1
            )

            DS_TRANSFORM = DS_FID_POINTS_NEW_FRAME @ np.linalg.inv(DS_FID_POINTS)

            self.TRANSFORMS[new_frame] = sp.interpolate.interp1d(
                downsample_time, DS_TRANSFORM, axis=0
            )(time)
            # self.TRANSFORMS[new_frame] /= np.abs(np.linalg.det(self.TRANSFORMS[new_frame]))[:, None, None]

            new_points = (self.TRANSFORMS[new_frame] @ _points[..., None])[..., 0]

            new_phi = np.arctan2(new_points[..., 1], new_points[..., 0]) % (2 * np.pi)
            new_theta = np.arcsin(new_points[..., 2])

            setattr(self, FRAMES[new_frame]["phi"], new_phi.reshape(np.shape(phi)))
            setattr(
                self, FRAMES[new_frame]["theta"], new_theta.reshape(np.shape(theta))
            )

            center_phi, center_theta = get_center_phi_theta(new_phi, new_theta)

            setattr(self, f'center_{FRAMES[new_frame]["phi"]}', center_phi)
            setattr(self, f'center_{FRAMES[new_frame]["theta"]}', center_theta)

    def offsets(self, frame, center, units="radians"):
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


# def dx_dy_to_phi_theta(dx, dy, center_phi, center_theta):
#     """
#     Convert array offsets to e.g. az and el.
#     """
#     # Face north, and look up at the zenith. Here, the translation is
#     #
#     #

#     input_shape = np.shape(dx)
#     _dx, _dy = np.atleast_1d(dx).ravel(), np.atleast_1d(dy).ravel()

#     # if we're looking at phi=0, theta=pi/2, then we have:
#     phi = np.arctan2(_dx, -_dy)
#     theta = np.pi / 2 - np.sqrt(_dx**2 + _dy**2)

#     x = np.cos(phi) * np.cos(theta)
#     y = np.sin(phi) * np.cos(theta)
#     z = np.sin(theta)

#     points = np.c_[x, y, z].T
#     points = get_rotation_matrix_3d(angles=np.pi / 2 - center_theta, axis=1) @ points
#     points = get_rotation_matrix_3d(angles=-center_phi, axis=2) @ points

#     new_phi = np.arctan2(points[1], points[0]) % (2 * np.pi)
#     new_theta = np.arcsin(points[2])

#     return new_phi.reshape(input_shape), new_theta.reshape(input_shape)


# def phi_theta_to_dx_dy(phi, theta, center_phi, center_theta):
#     """
#     This is the inverse of the other one.
#     """
#     # Face north, and look up at the zenith. Here, the translation is
#     input_shape = np.shape(phi)
#     _phi, _theta = np.atleast_1d(phi).ravel(), np.atleast_1d(theta).ravel()

#     x = np.cos(_phi) * np.cos(_theta)
#     y = np.sin(_phi) * np.cos(_theta)
#     z = np.sin(_theta)

#     points = np.c_[x, y, z].T
#     points = get_rotation_matrix_3d(angles=center_phi, axis=2) @ points
#     points = get_rotation_matrix_3d(angles=center_theta - np.pi / 2, axis=1) @ points

#     p = np.angle(points[0] + 1j * points[1])
#     r = np.arccos(points[2])

#     return (r * np.sin(p)).reshape(input_shape), (-r * np.cos(p)).reshape(input_shape)
