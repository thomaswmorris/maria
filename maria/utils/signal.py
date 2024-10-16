import numpy as np
import scipy as sp


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


def decompose(DATA, mode="us", downsample_rate=1):
    downsampled_data = downsample(DATA, rate=downsample_rate, method="triangle")
    rms = downsampled_data.std(axis=-1)
    u, s, v = np.linalg.svd(downsampled_data / rms[:, None], full_matrices=False)
    uv_norm = v.std(axis=-1) * np.sign(u.mean(axis=0))
    s_norm = np.sqrt(np.square(s).sum())
    u *= s_norm * rms[:, None] * uv_norm[None, :]
    s /= s_norm

    if mode == "us":
        return u, s
    if mode == "uv":
        return np.matmul(u, np.diag(s)), np.matmul(
            np.linalg.pinv(np.matmul(u, np.diag(s))), DATA
        )
    if mode == "usv":
        return u, s, np.matmul(np.linalg.pinv(np.matmul(u, np.diag(s))), DATA)


def get_bspline_basis(x, spacing=60, order=3, **kwargs):
    k = np.arange(x[0], x[-1], spacing)
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
            B[i, j] = B[i - 1, j] * (x - t[j]) / (t[j + i] - t[j]) + B[i - 1, j + 1] * (
                t[j + i + 1] - x
            ) / (t[j + i + 1] - t[j + 1])

    basis = B[-1]  # .reshape(-1, len(x))
    basis = basis[basis.sum(axis=-1) > 0]

    return basis


def detrend(D, order=3):
    x = np.linspace(-1, 1, D.shape[-1])
    X = np.c_[[x**i for i in range(order + 1)]]
    A = D @ X.T @ np.linalg.inv(X @ X.T).T

    return D - A @ X


def lowpass(data, fc, sample_rate, order=1, axis=-1):
    sos = sp.signal.bessel(
        2 * (order + 1), 2 * fc / sample_rate, analog=False, btype="low", output="sos"
    )  # noqa
    return sp.signal.sosfilt(sos, data, axis=axis)


def highpass(data, fc, sample_rate, order=1, axis=-1):
    sos = sp.signal.bessel(
        2 * (order + 1), 2 * fc / sample_rate, analog=False, btype="high", output="sos"
    )  # noqa
    return sp.signal.sosfilt(sos, data, axis=axis)


def bandpass(data, f_lower, f_upper, sample_rate, order=1, axis=-1):
    kwargs = {"sample_rate": sample_rate, "order": order, "axis": axis}  # noqa
    return highpass(lowpass(data, f_upper, **kwargs), f_lower, **kwargs)  # noqa
