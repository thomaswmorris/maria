import astropy.constants as const
import numpy as np
from astropy import units as u

# Kelvin CMB to Jy/pixel
# ----------------------------------------------------------------------
global Tcmb
Tcmb = 2.7255


symbols = {"radians": "rad"}


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


def getJynorm():
    factor = 2e26
    factor *= (const.k_B * Tcmb * u.Kelvin) ** 3  # (kboltz*Tcmb)**3.0
    factor /= (const.h * const.c) ** 2  # (hplanck*clight)**2.0
    return factor.value


def getx(freq):
    factor = const.h * freq * u.Hz / const.k_B / (Tcmb * u.Kelvin)
    return factor.to(u.dimensionless_unscaled).value


def KcmbToJy(freq):
    x = getx(freq)
    factor = getJynorm() / Tcmb
    factor *= (x**4) * np.exp(x) / (np.expm1(x) ** 2)
    return factor


# Kelvin CMB to Kelvin brightness
# ----------------------------------------------------------------------
def KcmbToKbright(freq):
    x = getx(freq)
    return np.exp(x) * ((x / np.expm1(x)) ** 2)


# Kelvin brightness to Jy/pixel
# ----------------------------------------------------------------------
def KbrightToJyPix(freq, ipix, jpix=None):
    jpix = jpix or ipix
    return KcmbToJyPix(freq, ipix, jpix) / KcmbToKbright(freq)


# Kelvin CMB to Jy/pixel
# ----------------------------------------------------------------------
def KcmbToJyPix(freq, ipix, jpix=None):
    jpix = jpix or ipix
    x = getx(freq)
    factor = getJynorm() / Tcmb
    factor *= (x**4) * np.exp(x) / (np.expm1(x) ** 2)
    factor *= np.abs(ipix * jpix) * (np.pi / 1.8e2) * (np.pi / 1.8e2)
    return factor
