import time as ttime

import numpy as np

from .coordinates import Coordinates  # noqa
from .transforms import (  # noqa
    dx_dy_to_phi_theta,
    get_center_phi_theta,
    phi_theta_to_dx_dy,
    phi_theta_to_xyz,
    xyz_to_phi_theta,
)


def now():
    return ttime.time()


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
