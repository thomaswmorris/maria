def repr_dms(x):
    mnt, sec = divmod(abs(x) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return f"{int(deg)}°{int(mnt)}'{int(sec)}\""


def repr_lat_lon(lat, lon):
    lat_repr = repr_dms(lat) + ("N" if lat > 0 else "S")
    lon_repr = repr_dms(lon) + ("E" if lon > 0 else "W")
    return f"{lat_repr}, {lon_repr}"
