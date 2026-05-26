import numpy as np
import scipy as sp


def get_rotation_matrix_2d(a):
    G = np.array([[0, -1], [1, 0]])  # generator
    A = np.expand_dims(a, axis=(-1, -2))  # angles
    return sp.linalg.expm(A * G)


def get_rotation_matrix_3d(**rotations):
    """
    A list of tuples [(dim, angle), ...] where we successively rotate around dim by angle.
    """
    dims = {"x": 0, "y": 1, "z": 2}
    R = np.eye(3)
    for axis, angle in rotations.items():
        i, j = (index for dim, index in dims.items() if dim != axis)
        S = np.zeros((*np.shape(angle), 3, 3))
        S[..., i, j] = angle
        R = sp.linalg.expm(S - np.swapaxes(S, -2, -1)) @ R
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
            f"Bad shape for entries (for signature {signature} we expect len(entries) = {int(n_axes * (n_axes - 1) / 2)}.",
        )

    i, j = np.triu_indices(n=n_axes, k=1)
    S = np.zeros((n_dim, n_dim))
    S[axes[i], axes[j]] = entries
    return sp.linalg.expm(S - S.T)


def compute_aligning_transform(points, signature, axes=None, n_init: int = 16):
    """
    Find a transform for some (..., n_dim) array of points so that the volume over all but the first axis is minimized.
    """
    *input_shape, n_dim = points.shape
    axes = axes or tuple(range(1, n_dim))
    args = points.reshape(-1, n_dim)

    def loss(entries, *args):
        tp = args[0] @ get_orthogonal_transform(signature=signature, entries=entries)
        if n_dim > 2:
            ch = sp.spatial.ConvexHull(tp[..., 1:])
            return np.log(ch.volume)
        else:
            return np.log(np.ptp(tp[..., 1:]))

    n_axes = sum(signature)
    n_dof = int(n_axes * (n_axes - 1) / 2)
    x0_samples = np.random.standard_normal(size=(n_init, n_dof))
    best_index = np.argmin([loss(x0, args) for x0 in x0_samples])

    res = sp.optimize.minimize(
        loss,
        x0=x0_samples[best_index],
        args=args,
        tol=1e-6,
        method="SLSQP",
    )

    if not res.success:
        raise RuntimeError("Could not find optimal rotation.")

    return get_orthogonal_transform(signature=signature, entries=res.x)
