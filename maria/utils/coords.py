import numpy as np

from ..units import Quantity


def repr_dms(x):
    mnt, sec = divmod(abs(x) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return f"{int(deg)}°{int(mnt)}’{int(sec)}”"


def repr_hms(x):
    m, s = divmod(abs(x / 15) * 3600, 60)
    h, m = divmod(m, 60)
    return f"{int(h):>02}ʰ{int(m):>02}ᵐ{s:.02f}ˢ"


def repr_lat_lon(lat, lon):
    lat_repr = repr_dms(lat) + ("N" if lat > 0 else "S")
    lon_repr = repr_dms(lon) + ("E" if lon > 0 else "W")
    return f"{lat_repr} {lon_repr}"


def repr_phi_theta(phi, theta, frame):
    phi_deg = np.degrees(phi) % 360
    theta_deg = np.degrees(theta)
    if frame == "az_el":
        return (f"az: {phi_deg:.02f}°", f"el: {theta_deg:.02f}°")
    if frame == "ra_dec":
        return (f"ra: {repr_hms(phi_deg)}", f"dec: {repr_dms(theta_deg)}")
    if frame == "galactic":
        return (f"l: {phi_deg:.02f}°", f"b: {theta_deg:.02f}°")
