from __future__ import annotations

# nothing in here should import from other maria module (so no double dots!!)
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from .coords import *  # noqa
from .functions import *  # noqa
from .linalg import *  # noqa
from .plotting import *  # noqa
from .rotations import *  # noqa
from .rounding import *  # noqa
from .signal import *  # noqa
from .time import *  # noqa


@jax.jit
def regular_digitization(x, bins):
    dx = jnp.mean(jnp.gradient(bins)) if len(bins) > 1 else 1.0
    return ((x - (bins.min() - dx)) / dx).astype(int).clip(min=0, max=len(bins))


def unpack_implicit_slice(key, ndims):
    key = key if isinstance(key, tuple) else tuple(key)
    explicit_slices = []
    for s in key:
        if s == Ellipsis:
            for _ in range(ndims + 1 - len(key)):
                explicit_slices.append(slice(None, None, None))
        else:
            explicit_slices.append(s)

    while len(explicit_slices) < ndims:
        explicit_slices.append(slice(None, None, None))

    return tuple(explicit_slices)


def compute_diameter(points, lazy=False, MAX_SAMPLE_SIZE: int = 10000, jitter: float = 0.0) -> float:
    """
    Parameters
    ----------
    points : type
        An (..., n_dim) array of offsets of something.
    """

    *input_shape, n_dim = points.shape
    X = points.reshape(-1, n_dim)

    if jitter:
        X += jitter * np.ptp(X, axis=-1).max() * np.random.standard_normal(size=X.shape)

    dim_mask = np.ptp(X, axis=tuple(range(X.ndim - 1))) > 0
    if not dim_mask.any():
        return 0.0

    if len(X) < 16:
        i, j = np.triu_indices(n=len(X), k=1)
        return np.sqrt(np.sum(np.square(X[i] - X[j]), axis=-1).max())

    if lazy:
        X = X[
            np.random.choice(
                a=len(X),
                size=np.minimum(len(X), MAX_SAMPLE_SIZE),
                replace=False,
            )
        ]
    hull = sp.spatial.ConvexHull(X[:, dim_mask])
    vertices = hull.points[hull.vertices]
    i, j = np.triu_indices(len(vertices), k=1)

    return float(np.sqrt(np.max(np.square(vertices[i] - vertices[j]).sum(axis=-1))))
