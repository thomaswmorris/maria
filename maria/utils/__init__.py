# this is the junk drawer of functions
import time as ttime
from datetime import datetime

import numpy as np
import pytz
import scipy as sp
from scipy import spatial  # noqa

from . import linalg, signal  # noqa


def get_utc_day_hour(t):
    dt = datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc)
    return dt.hour + dt.minute / 60 + dt.second / 3600


def get_utc_year_day(t):
    tt = datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc).timetuple()
    return tt.tm_yday + get_utc_day_hour(t) / 24 - 1


def get_utc_year(t):
    return datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc).year


def now():
    return ttime.time()


def repr_dms(x):
    mnt, sec = divmod(abs(x) * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return f"{int(deg)}°{int(mnt)}'{int(sec)}\""


def repr_lat_lon(lat, lon):
    lat_repr = repr_dms(lat) + ("N" if lat > 0 else "S")
    lon_repr = repr_dms(lon) + ("E" if lon > 0 else "W")
    return f"{lat_repr}, {lon_repr}"


def lazy_diameter(offsets, MAX_SAMPLE_SIZE: int = 10000) -> float:
    """
    Parameters
    ----------
    offsets : type
        An (..., n_dim) array of offsets of something.
    """

    *input_shape, n_dim = offsets.shape
    X = offsets.reshape(-1, n_dim)

    dim_mask = np.ptp(X, axis=tuple(range(X.ndim - 1))) > 0
    if not dim_mask.any():
        return 0.0

    subset_index = np.random.choice(
        a=len(X), size=np.minimum(len(X), MAX_SAMPLE_SIZE), replace=False
    )
    hull = sp.spatial.ConvexHull(X[subset_index][:, dim_mask])
    vertices = hull.points[hull.vertices]
    i, j = np.triu_indices(len(vertices), k=1)

    return float(np.sqrt(np.max(np.square(vertices[i] - vertices[j]).sum(axis=-1))))
