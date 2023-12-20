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


def get_center_phi_theta(phi, theta):
    """ """
    xyz = phi_theta_to_xyz(phi, theta)

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


# class Coordinator:
#     # what three-dimensional rotation matrix takes (frame 1) to (frame 2) ?
#     # we use astropy to compute this for a few test points, and then use the
#     # answer it to efficiently broadcast very big arrays

#     def __init__(self, lon, lat):
#         self.location = ap.coordinates.EarthLocation.from_geodetic(lon=lon, lat=lat)

#         self.fid_p = np.radians(np.array([0, 0, 90]))
#         self.fid_t = np.radians(np.array([90, 0, 0]))
#         self.fid_xyz = np.c_[
#             np.sin(self.fid_p) * np.cos(self.fid_t),
#             np.cos(self.fid_p) * np.cos(self.fid_t),
#             np.sin(self.fid_t),
#         ]  # the XYZ coordinates of our fiducial test points on the unit sphere

#         # in order for this to be efficient, we need to use time-invariant frames

#         # you are standing a the north pole looking toward lon = -90 (+x)
#         # you are standing a the north pole looking toward lon = 0 (+y)
#         # you are standing a the north pole looking up (+z)

#     def transform(self, unix, phi, theta, in_frame, out_frame):
#         _unix = np.atleast_2d(unix).copy()
#         _phi = np.atleast_2d(phi).copy()
#         _theta = np.atleast_2d(theta).copy()

#         if not _phi.shape == _theta.shape:
#             raise ValueError("'phi' and 'theta' must be the same shape")
#         if not 1 <= len(_phi.shape) == len(_theta.shape) <= 2:
#             raise ValueError("'phi' and 'theta' must be either 1- or 2-dimensional")
#         if not unix.shape[-1] == _phi.shape[-1] == _theta.shape[-1]:
#             ("'unix', 'phi' and 'theta' must have the same shape in their last axis")

#         epoch = _unix.mean()
#         obstime = ap.time.Time(epoch, format="unix")
#         rad = ap.units.rad

#         if in_frame == "az_el":
#             self.c = ap.coordinates.SkyCoord(
#                 az=self.fid_p * rad,
#                 alt=self.fid_t * rad,
#                 obstime=obstime,
#                 frame="altaz",
#                 location=self.location,
#             )
#         if in_frame == "ra_dec":
#             self.c = ap.coordinates.SkyCoord(
#                 ra=self.fid_p * rad,
#                 dec=self.fid_t * rad,
#                 obstime=obstime,
#                 frame="icrs",
#                 location=self.location,
#             )
#         # if in_frame == 'galactic':
#         # self.c = ap.coordinates.SkyCoord(l  = self.fid_p * rad, b   = self.fid_t * rad, obstime = ot,
#         # frame = 'galactic', location = self.location)

#         if out_frame == "ra_dec":
#             self._c = self.c.icrs
#             self.rot_p, self.rot_t = self._c.ra.rad, self._c.dec.rad
#         if out_frame == "az_el":
#             self._c = self.c.altaz
#             self.rot_p, self.rot_t = self._c.az.rad, self._c.alt.rad

#         # if out_frame == 'galactic': self._c = self.c.galactic; self.rot_p, self.rot_t = self._c.l.rad,  self._c.b.rad

#         self.rot_xyz = np.c_[
#             np.sin(self.rot_p) * np.cos(self.rot_t),
#             np.cos(self.rot_p) * np.cos(self.rot_t),
#             np.sin(self.rot_t),
#         ]  # the XYZ coordinates of our rotated test points on the unit sphere

#         self.R = np.linalg.lstsq(self.fid_xyz, self.rot_xyz, rcond=-1)[
#             0
#         ]  # what matrix takes us (fid_xyz -> rot_xyz)?

#         if (in_frame, out_frame) == ("ra_dec", "az_el"):
#             _phi -= (_unix - epoch) * (2 * np.pi / 86163.0905)

#         trans_xyz = np.swapaxes(
#             np.matmul(
#                 np.swapaxes(
#                     np.concatenate(
#                         [
#                             (np.sin(_phi) * np.cos(_theta))[None],
#                             (np.cos(_phi) * np.cos(_theta))[None],
#                             np.sin(_theta)[None],
#                         ],
#                         axis=0,
#                     ),
#                     0,
#                     -1,
#                 ),
#                 self.R,
#             ),
#             0,
#             -1,
#         )

#         trans_phi, trans_theta = np.arctan2(trans_xyz[0], trans_xyz[1]), np.arcsin(
#             trans_xyz[2]
#         )

#         if (in_frame, out_frame) == ("az_el", "ra_dec"):
#             trans_phi += (_unix - epoch) * (2 * np.pi / 86163.0905)

#         return np.reshape(trans_phi % (2 * np.pi), phi.shape), np.reshape(
#             trans_theta, theta.shape
#         )
