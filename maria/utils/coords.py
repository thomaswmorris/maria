import numpy as np


def dms_to_deg(d: float = 0, m: float = 0, s: float = 0):
    return np.radians(d + m / 60 + s / 3600)


def hms_to_deg(h: float = 0, m: float = 0, s: float = 0):
    return np.radians(h * 15 + m * 15 / 60 + s * 15 / 3600)


def deg_to_signed_dms(x: float):
    sign = -1 if x < 0 else 1
    mnt, sec = divmod(abs(x) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return int(sign), int(deg), int(mnt), sec


def deg_to_signed_hms(x: float):
    sign = -1 if x < 0 else 1
    mnt, sec = divmod(abs(x) * 3600 / 15, 60)
    deg, mnt = divmod(mnt, 60)
    return int(sign), int(deg), int(mnt), sec


def repr_dms(x: float):
    sign, d, m, s = deg_to_signed_dms(x)
    return f"{int(sign * d):>02}°{int(m):>02}’{s:.02f}”"


def repr_hms(x: float):
    sign, h, m, s = deg_to_signed_hms(x)
    return f"{int(sign * h):>02}ʰ{int(m):>02}ᵐ{s:.02f}ˢ"


def repr_lat_lon(lat, lon):
    lat_repr = repr_dms(abs(lat)) + ("N" if lat > 0 else "S")
    lon_repr = repr_dms(abs(lon)) + ("E" if lon > 0 else "W")
    return f"{lat_repr} {lon_repr}"


def repr_phi_theta(phi, theta, frame):
    phi_deg = np.degrees(phi % 360)
    theta_deg = np.degrees(theta)
    if frame == "az_el":
        return (f"az: {phi_deg:.02f}°", f"el: {theta_deg:.02f}°")
    if frame == "ra_dec":
        return (f"ra: {repr_hms(phi_deg)}", f"dec: {repr_dms(theta_deg)}")
    if frame == "galactic":
        return (f"l: {phi_deg:.02f}°", f"b: {theta_deg:.02f}°")
