import numpy as np


def repr_dms(x):
    sign = -1 if x < 0 else 1
    mnt, sec = divmod(np.degrees(abs(x)) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return f"{int(sign * deg)}°{int(mnt)}’{int(sec)}”"


def repr_hms(x):
    sign = -1 if x < 0 else 1
    m, s = divmod(np.degrees(abs(x)) / 15 * 3600, 60)
    h, m = divmod(m, 60)
    return f"{int(sign * h):>02}ʰ{int(m):>02}ᵐ{s:.02f}ˢ"


def repr_lat_lon(lat, lon):
    lat_repr = repr_dms(abs(lat)) + ("N" if lat > 0 else "S")
    lon_repr = repr_dms(abs(lon)) + ("E" if lon > 0 else "W")
    return f"{lat_repr} {lon_repr}"


def repr_phi_theta(phi, theta, frame):
    phi_deg = phi % 360
    theta_deg = theta
    if frame == "az_el":
        return (f"az: {phi_deg:.02f}°", f"el: {theta_deg:.02f}°")
    if frame == "ra_dec":
        return (f"ra: {repr_hms(phi_deg)}", f"dec: {repr_dms(theta_deg)}")
    if frame == "galactic":
        return (f"l: {phi_deg:.02f}°", f"b: {theta_deg:.02f}°")
