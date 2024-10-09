import numpy as np


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

    def __float__(self):
        return self.rad

    def __repr__(self):
        units = self.units
        if units == "arcsec":
            return f"{round(self.arcsec, 2)}”"
        if units == "arcmin":
            return f"{round(self.arcmin, 2)}’"
        if units == "degrees":
            return f"{round(self.deg, 2)}°"

    @property
    def units(self):
        # peak-to-peak
        max_deg = self.deg if self.is_scalar else self.deg.max()

        if max_deg < 0.5 / 60:
            return "arcsec"
        if max_deg < 0.5:
            return "arcmin"

        return "degrees"
