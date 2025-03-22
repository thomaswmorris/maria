from __future__ import annotations

import os

import astropy.constants as const
import numpy as np
from astropy import units as u

from ..constants import T_CMB, c, g, k_B  # noqa
from .angle import Angle  # noqa
from .prefixes import PREFIXES  # noqa
from .quantities import QUANTITIES, UNITS, Quantity, parse_units  # noqa

here, this_filename = os.path.split(__file__)
symbols = {"radians": "rad"}


def getJynorm():
    factor = 2e26
    factor *= (const.k_B * T_CMB * u.Kelvin) ** 3  # (kboltz*T_CMB)**3.0
    factor /= (const.h * const.c) ** 2  # (hplanck*clight)**2.0
    return factor.value


def getx(freq):
    factor = const.h * freq * u.Hz / const.k_B / (T_CMB * u.Kelvin)
    return factor.to(u.dimensionless_unscaled).value


def KcmbToJy(freq):
    x = getx(freq)
    factor = getJynorm() / T_CMB
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
    factor = getJynorm() / T_CMB
    factor *= (x**4) * np.exp(x) / (np.expm1(x) ** 2)
    factor *= np.abs(ipix * jpix) * (np.pi / 1.8e2) * (np.pi / 1.8e2)
    return factor
