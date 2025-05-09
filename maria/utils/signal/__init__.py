from __future__ import annotations

import numpy as np
import scipy as sp

from .filters import bandpass, highpass, lowpass  # noqa


def get_kernel(n, kind="triangle"):
    if kind == "triangle":
        kernel = np.r_[np.linspace(0, 1, n + 1)[1:], np.linspace(1, 0, n + 1)[1:-1]]

        return kernel / kernel.sum()


def fast_downsample(DATA, r=1):
    *in_shape, n = DATA.shape
    CUMSUM = np.cumsum(np.atleast_2d(DATA), axis=-1)
    return (CUMSUM[..., r::r] - CUMSUM[..., :-r:r]).reshape(*in_shape, -1) / r


def downsample(DATA, rate, axis=-1, method=None):
    if method is None:
        return np.swapaxes(np.swapaxes(DATA, 0, axis)[::rate], 0, axis)

    if method == "fast":
        return fast_downsample(
            DATA,
            r=rate,
        )

    if method == "flat":
        _DATA = np.swapaxes(DATA, 0, axis)
        cs_data = np.cumsum(_DATA, axis=0)
        return np.swapaxes((cs_data[rate::rate] - cs_data[:-rate:rate]) / rate, 0, axis)

    if method == "triangle":
        if rate == 1:
            return DATA
        if rate < 1:
            raise ValueError("downsample rate must be an integer >= 1")

        _DATA = np.swapaxes(DATA, 0, axis)
        kernel = np.expand_dims(
            get_kernel(n=rate, kind="triangle"),
            axis=tuple(np.arange(1, len(DATA.shape))),
        )
        n_kern = len(kernel)
        starts = np.arange(0, len(_DATA) - n_kern, rate)
        ends = starts + n_kern

        return np.swapaxes(
            np.r_[[np.sum(_DATA[s:e] * kernel, axis=0) for s, e in zip(starts, ends)]],
            0,
            axis,
        )


def decompose(data, k: int = None):
    k = k or len(data)

    if data.ndim != 2:
        raise ValueError("'data' must be a 2D array.")

    if k < len(data):
        u, s, vh = sp.sparse.linalg.svds(data, k=k, which="LM")
    else:
        u, s, vh = sp.linalg.svd(data, full_matrices=False)

    norm = np.sign(u.mean(axis=0)) * vh.std(axis=-1)
    a = u @ np.diag(s * norm)
    b = np.diag(1 / norm) @ vh

    mode_sort = np.argsort(-s)

    return a[:, mode_sort], b[mode_sort]


def spline(x: float, y: float, spacing: float, order: int = 3):
    # this has shape (n_basis, len(x))
    B = bspline_basis(x, spacing=spacing, order=order)

    A = B @ B.T


def bspline_basis(x, spacing, order=3):
    k = np.arange(np.min(x), np.max(x), spacing)
    if not len(k) > 0:
        k = np.array([np.mean(x)])
    k += x.mean() - k.mean()
    t = np.r_[
        k[0] + spacing * np.arange(-order - 1, 0),
        k,
        k[-1] + spacing * np.arange(1, order + 2),
    ]

    B = np.zeros((order + 1, len(t) + 1, len(x)))
    B[0, np.digitize(x, t) - 1, np.arange(len(x))] = 1

    for i in range(1, order + 1):
        for j in range(len(t) - (order + 1)):
            B[i, j] = B[i - 1, j] * (x - t[j]) / (t[j + i] - t[j]) + B[i - 1, j + 1] * (t[j + i + 1] - x) / (
                t[j + i + 1] - t[j + 1]
            )

    basis = B[-1]  # .reshape(-1, len(x))
    basis = basis[basis.sum(axis=-1) > 0]

    total_weight_per_mode = basis.sum(axis=1)
    basis = basis[total_weight_per_mode > 0.1 * total_weight_per_mode.max()]

    return basis / basis.sum(axis=0)


def cross_basis(X: list, spacing: list, order: list):
    basis = np.ones((1, 1))

    for dim, x in enumerate(X):
        x_basis = bspline_basis(x, spacing[dim], order[dim])
        basis = (x_basis[:, None] * basis).reshape(-1, len(x))
        basis = basis[basis.sum(axis=-1) > 0]

    return basis


def detrend(D, order=3):
    x = np.linspace(-1, 1, D.shape[-1])
    X = np.c_[[x**i for i in range(order + 1)]]
    A = D @ X.T @ np.linalg.inv(X @ X.T).T

    return D - A @ X


def remove_slope(D):
    return D - np.linspace(D[..., 0], D[..., -1], D.shape[-1]).T
