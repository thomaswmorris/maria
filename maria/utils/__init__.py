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


def generate_power_law_noise(n: tuple = (256, 256), cutoff=1e0, beta=None):
    ndim = len(n)  # noqa
    beta = beta or 8 / 6

    X_list = np.meshgrid(*[np.linspace(0, 1, _n) for _n in n])
    K_list = np.meshgrid(*[np.fft.fftfreq(_n, d=1 / _n) for _n in n])

    P = (cutoff**2 + sum([K**2 for K in K_list])) ** -(beta / 2)

    F = np.real(np.fft.fftn(P * np.fft.ifftn(np.random.standard_normal(size=n))))

    return *X_list, (F - F.mean()) / F.std()


def compute_diameter(points, lazy=False, MAX_SAMPLE_SIZE: int = 10000) -> float:
    """
    Parameters
    ----------
    points : type
        An (..., n_dim) array of offsets of something.
    """

    *input_shape, n_dim = points.shape
    X = points.reshape(-1, n_dim)

    dim_mask = np.ptp(X, axis=tuple(range(X.ndim - 1))) > 0
    if not dim_mask.any():
        return 0.0

    if lazy:
        X = X[
            np.random.choice(
                a=len(X), size=np.minimum(len(X), MAX_SAMPLE_SIZE), replace=False
            )
        ]
    hull = sp.spatial.ConvexHull(X[:, dim_mask])
    vertices = hull.points[hull.vertices]
    i, j = np.triu_indices(len(vertices), k=1)

    return float(np.sqrt(np.max(np.square(vertices[i] - vertices[j]).sum(axis=-1))))


def get_rotation_matrix(**rotations):
    """
    A list of tuples [(dim, angle), ...] where we successively rotate around dim by angle.
    """
    dims = {"x": 0, "y": 1, "z": 2}
    R = np.eye(3)
    for axis, angle in rotations.items():
        i, j = [index for dim, index in dims.items() if dim != axis]
        S = np.zeros((3, 3))
        S[i, j] = angle
        R = sp.linalg.expm(S - S.T) @ R
    return R


def compute_optimal_rotation(points, axes=None):
    """
    Find a rotation for some points so that the projection onto the last two axes has the minimum area.
    """

    *input_shape, ndim = points.shape
    axes = axes or tuple(range(ndim))

    def loss(a, *args):
        tp = args[0] @ get_rotation_matrix(z=a[0])
        ch = sp.spatial.ConvexHull(tp[:, 1:])
        return ch.volume

    res = sp.optimize.minimize(loss, x0=[0], args=points, tol=1e-10, method="SLSQP")

    if not res.success:
        raise RuntimeError("Could not find optimal rotation.")

    return get_rotation_matrix(z=res.x[0])
