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


def decompose(D, k=64, batch=True):
    *batch_shape, n_d, n_t = D.shape
    dnorm = D.std(axis=-1)

    if batch_shape:
        if batch:
            A_list, B_list = [], []
            for d in D.reshape(-1, n_d, n_t):
                A, B = decompose(d)
                A_list.append(A)
                B_list.append(B)
            return (
                np.stack(A_list, axis=0).reshape(*batch_shape, n_d, k),
                np.stack(B_list, axis=0).reshape(*batch_shape, k, n_t),
            )

        else:
            A, B = decompose(D.reshape(-1, n_t))
            return A.reshape(*batch_shape, n_d, -1), B

    dnorm = np.sqrt(np.sum(np.square(D), axis=-1))
    u, s, v = sp.sparse.linalg.svds(D / dnorm[..., None], k=k)
    vnorm = np.sqrt(np.sum(np.square(v), axis=-1))
    mode_sort = np.argsort(-s)
    return (dnorm[:, None] * u * s * vnorm)[..., mode_sort], (v / vnorm[..., None])[mode_sort]


def bspline_knots(t, spacing, order):
    tmin = t.min()
    tmax = t.max()
    n_bins = int(np.maximum((tmax - tmin) // spacing, 1))  # how many bins straddle the data?
    n_basis = n_bins + order + 1

    k = spacing * np.arange(n_bins, dtype=float)  # so that it straddles the whole domain
    k += float(tmax + tmin) / 2 - k.mean()
    k = np.r_[
        k[0] + spacing * np.arange(-order - 1, 0),
        k,
        k[-1] + spacing * np.arange(1, order + 2),
    ]  # pad the edges

    return k


def bspline_basis_from_knots(t, k, order):
    n_basis = len(k) - order - 1
    B = np.zeros((len(k) + 1, order + 1, len(t)))
    B[np.digitize(t, k) - 1, 0, np.arange(len(t))] = 1

    for p in range(1, order + 1):
        for i in range(len(k) - p - 1):
            B[i, p] = B[i, p - 1] * (t - k[i]) / (k[i + p] - k[i]) + B[i + 1, p - 1] * (k[i + p + 1] - t) / (
                k[i + p + 1] - k[i + 1]
            )

    return B[:n_basis, -1]


def bspline_basis(t, spacing, order):
    k = bspline_knots(t, spacing, order)
    return bspline_basis_from_knots(t, k, order)


def fit_bspline(y, x, spacing, order=3):
    B = bspline_basis(x, spacing=spacing, order=order)
    A = y @ (np.linalg.inv(B @ B.T) @ B).T
    return A @ B


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


def grouper(iterable, min_length=1, max_length=np.inf, overlap=False):
    start = np.inf
    prev_value = False
    for index, this_value in enumerate(iterable):
        if this_value:
            if not prev_value:
                start = index
            if prev_value:
                if index - start >= max_length:
                    yield (start, index)
                    start = index
        if not this_value:
            if prev_value:
                if index - start >= min_length:
                    yield (start, index)
        prev_value = this_value

    if prev_value:
        yield (start, index + 1)


# print(list(grouper([True, True, False, True, True])))
# print(list(grouper([False, True, True, True, True, True, False, True, True], max_length=10, overlap=True)))
# print(list(grouper([False, True, True, True, True, True, False, True, True], max_length=10, overlap=False)))
