import numpy as np


def dms_to_deg(d: float = 0, m: float = 0, s: float = 0):
    return np.radians(d + m / 60 + s / 3600)


def hms_to_deg(h: float = 0, m: float = 0, s: float = 0):
    return np.radians(h * 15 + m * 15 / 60 + s * 15 / 3600)


def deg_to_signed_dms(x: float, precision: int = 6):
    x = round(x, precision)
    sign = -1 if x < 0 else 1
    mnt, sec = divmod(abs(x) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return int(sign), int(deg), int(mnt), sec


def deg_to_signed_hms(x: float, precision: int = 6):
    x = round(x, precision)
    sign = -1 if x < 0 else 1
    mnt, sec = divmod(abs(x) * 3600 / 15, 60)
    deg, mnt = divmod(mnt, 60)
    return int(sign), int(deg), int(mnt), sec
