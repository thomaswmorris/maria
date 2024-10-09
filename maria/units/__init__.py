import os

import astropy.constants as const
import numpy as np
import pandas as pd
from astropy import units as u

from .angle import Angle  # noqa
from .constants import T_CMB

# Kelvin CMB to Jy/pixel
# ----------------------------------------------------------------------
symbols = {"radians": "rad"}

here, this_filename = os.path.split(__file__)

BASE_TOD_UNITS = ["W", "K_RJ", "K_CMB"]
prefixes = pd.read_csv(f"{here}/prefixes.csv", index_col=0)


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
