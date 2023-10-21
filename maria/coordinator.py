import astropy as ap
import numpy as np


class Coordinator:
    # what three-dimensional rotation matrix takes (frame 1) to (frame 2) ?
    # we use astropy to compute this for a few test points, and then use the
    # answer it to efficiently broadcast very big arrays

    def __init__(self, lon, lat):
        self.location = ap.coordinates.EarthLocation.from_geodetic(lon=lon, lat=lat)

        self.fid_p = np.radians(np.array([0, 0, 90]))
        self.fid_t = np.radians(np.array([90, 0, 0]))
        self.fid_xyz = np.c_[
            np.sin(self.fid_p) * np.cos(self.fid_t),
            np.cos(self.fid_p) * np.cos(self.fid_t),
            np.sin(self.fid_t),
        ]  # the XYZ coordinates of our fiducial test points on the unit sphere

        # in order for this to be efficient, we need to use time-invariant frames

        # you are standing a the north pole looking toward lon = -90 (+x)
        # you are standing a the north pole looking toward lon = 0 (+y)
        # you are standing a the north pole looking up (+z)

    def transform(self, unix, phi, theta, in_frame, out_frame):
        _unix = np.atleast_2d(unix).copy()
        _phi = np.atleast_2d(phi).copy()
        _theta = np.atleast_2d(theta).copy()

        if not _phi.shape == _theta.shape:
            raise ValueError("'phi' and 'theta' must be the same shape")
        if not 1 <= len(_phi.shape) == len(_theta.shape) <= 2:
            raise ValueError("'phi' and 'theta' must be either 1- or 2-dimensional")
        if not unix.shape[-1] == _phi.shape[-1] == _theta.shape[-1]:
            ("'unix', 'phi' and 'theta' must have the same shape in their last axis")

        epoch = _unix.mean()
        obstime = ap.time.Time(epoch, format="unix")
        rad = ap.units.rad

        if in_frame == "az_el":
            self.c = ap.coordinates.SkyCoord(
                az=self.fid_p * rad,
                alt=self.fid_t * rad,
                obstime=obstime,
                frame="altaz",
                location=self.location,
            )
        if in_frame == "ra_dec":
            self.c = ap.coordinates.SkyCoord(
                ra=self.fid_p * rad,
                dec=self.fid_t * rad,
                obstime=obstime,
                frame="icrs",
                location=self.location,
            )
        # if in_frame == 'galactic':
        # self.c = ap.coordinates.SkyCoord(l  = self.fid_p * rad, b   = self.fid_t * rad, obstime = ot,
        # frame = 'galactic', location = self.location)

        if out_frame == "ra_dec":
            self._c = self.c.icrs
            self.rot_p, self.rot_t = self._c.ra.rad, self._c.dec.rad
        if out_frame == "az_el":
            self._c = self.c.altaz
            self.rot_p, self.rot_t = self._c.az.rad, self._c.alt.rad

        # if out_frame == 'galactic': self._c = self.c.galactic; self.rot_p, self.rot_t = self._c.l.rad,  self._c.b.rad

        self.rot_xyz = np.c_[
            np.sin(self.rot_p) * np.cos(self.rot_t),
            np.cos(self.rot_p) * np.cos(self.rot_t),
            np.sin(self.rot_t),
        ]  # the XYZ coordinates of our rotated test points on the unit sphere

        self.R = np.linalg.lstsq(self.fid_xyz, self.rot_xyz, rcond=-1)[
            0
        ]  # what matrix takes us (fid_xyz -> rot_xyz)?

        if (in_frame, out_frame) == ("ra_dec", "az_el"):
            _phi -= (_unix - epoch) * (2 * np.pi / 86163.0905)

        trans_xyz = np.swapaxes(
            np.matmul(
                np.swapaxes(
                    np.concatenate(
                        [
                            (np.sin(_phi) * np.cos(_theta))[None],
                            (np.cos(_phi) * np.cos(_theta))[None],
                            np.sin(_theta)[None],
                        ],
                        axis=0,
                    ),
                    0,
                    -1,
                ),
                self.R,
            ),
            0,
            -1,
        )

        trans_phi, trans_theta = np.arctan2(trans_xyz[0], trans_xyz[1]), np.arcsin(
            trans_xyz[2]
        )

        if (in_frame, out_frame) == ("az_el", "ra_dec"):
            trans_phi += (_unix - epoch) * (2 * np.pi / 86163.0905)

        return np.reshape(trans_phi % (2 * np.pi), phi.shape), np.reshape(
            trans_theta, theta.shape
        )
