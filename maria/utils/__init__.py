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


# def time():
#     global reference_time
#     duration = ttime.monotonic() - ref_time
#     logger.info(f"Initialized atmosphere in {int(1e3 * duration)} ms.")
#     ref_time = ttime.monotonic()


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


def get_orthogonal_transform(signature, entries):
    """
    A list of tuples [(dim, angle), ...] where we successively rotate around dim by angle.
    """

    axes = np.where(signature)[0]
    n_dim = len(signature)
    n_axes = sum(signature)

    if n_axes * (n_axes - 1) / 2 != len(entries):
        raise ValueError(
            f"Bad shape for entries (for signature {signature} we expect len(entries) = {int(n_axes * (n_axes - 1) / 2)}."
        )

    i, j = np.triu_indices(n=n_axes, k=1)
    S = np.zeros((n_dim, n_dim))
    S[axes[i], axes[j]] = entries
    return sp.linalg.expm(S - S.T)


def compute_aligning_transform(points, signature, axes=None):
    """
    Find a transform for some (..., n_dim) array of points so that the volume over all but the first axis is minimized.
    """
    *input_shape, n_dim = points.shape
    axes = axes or tuple(range(1, n_dim))

    def loss(entries, *args):
        tp = args[0] @ get_orthogonal_transform(signature=signature, entries=entries)
        if n_dim > 2:
            ch = sp.spatial.ConvexHull(tp[..., 1:])
            return np.log(ch.volume)
        else:
            return np.ptp(tp[..., 1:])

    n_axes = sum(signature)
    res = sp.optimize.minimize(
        loss,
        x0=np.zeros(int(n_axes * (n_axes - 1) / 2)),
        args=points.reshape(-1, n_dim),
        tol=1e-10,
        method="SLSQP",
    )

    if not res.success:
        raise RuntimeError("Could not find optimal rotation.")

    return get_orthogonal_transform(signature=signature, entries=res.x)
