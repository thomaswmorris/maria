import numpy as np
import scipy as sp


def weighted_binned_mean(x, y, bins, ignore_nan=True, weights=None):
    if weights is None:
        weights = np.ones(x.shape)

    numer = sp.stats.binned_statistic(x, weights * y, bins=bins, statistic="sum")[0]
    denom = sp.stats.binned_statistic(x, weights, bins=bins, statistic="sum")[0]

    return numer / denom


def get_kernel(n, kind="triangle"):
    if kind == "triangle":
        kernel = np.r_[np.linspace(0, 1, n + 1)[1:], np.linspace(1, 0, n + 1)[1:-1]]

        return kernel / kernel.sum()


def unwrap_angle(angle):
    mean_angle = np.angle(np.exp(1j * angle).mean())
    adju_angle = np.angle(np.exp(1j * (angle - mean_angle))) + mean_angle
    cntr_angle = 0.5 * (adju_angle.max() + adju_angle.min())
    return np.angle(np.exp(1j * (angle - cntr_angle))) + cntr_angle


def grouper(iterable, tol=1):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= tol:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def downsample(DATA, rate, axis=-1, method="triangle"):
    if method == "flat":
        _DATA = np.swapaxes(DATA, 0, axis)
        cs_data = np.cumsum(_DATA, axis=0)
        return np.swapaxes((cs_data[rate::rate] - cs_data[:-rate:rate]) / rate, 0, axis)

    else:
        if rate == 1:
            return DATA
        if rate < 1:
            raise ValueError("downsample rate must be an integer >= 1")

        _DATA = np.swapaxes(DATA, 0, axis)
        kernel = np.expand_dims(
            get_kernel(n=rate, kind=method), axis=tuple(np.arange(1, len(DATA.shape)))
        )
        n_kern = len(kernel)
        starts = np.arange(0, len(_DATA) - n_kern, rate)
        ends = starts + n_kern

        return np.swapaxes(
            np.r_[[np.sum(_DATA[s:e] * kernel, axis=0) for s, e in zip(starts, ends)]],
            0,
            axis,
        )


def get_phase_template(DATA, phase, n_phase_bins, discriminator=None):
    if discriminator is None:
        discriminator = np.ones(DATA.shape[0])

    nd, nt = DATA.shape
    TEMPLATE = np.zeros((nd, nt))

    for ud in np.unique(discriminator):
        mask = discriminator == ud
        D = DATA[mask].copy()
        D_mean = D.mean(axis=0)

        fractional_bin_index = phase * (n_phase_bins / (2 * np.pi))

        from sklearn.preprocessing import PolynomialFeatures

        template_degree = 2
        poly = PolynomialFeatures(degree=template_degree).fit_transform(
            np.linspace(-1, 1, nt)[:, None]
        )

        P = np.zeros((nt, n_phase_bins))
        P[np.arange(nt), np.floor(fractional_bin_index).astype(int) % n_phase_bins] = (
            1 - fractional_bin_index % 1
        )
        P[np.arange(nt), np.ceil(fractional_bin_index).astype(int) % n_phase_bins] = (
            fractional_bin_index % 1
        )

        P = sp.ndimage.gaussian_filter1d(P, sigma=1, axis=1, mode="wrap")
        PP = np.concatenate(
            [P * poly[:, i][:, None] for i in range(template_degree + 1)], axis=1
        )
        PD = np.matmul(np.linalg.inv(np.matmul(PP.T, PP)), np.matmul(PP.T, D_mean))
        template = np.matmul(PP, PD)

        gains = np.sum(template * D, axis=1) / np.square(template).sum()
        TEMPLATE[mask] = np.outer(gains, template)

    return TEMPLATE


def make_cuts(D, n_filt=3, downsample_rate=4, max_cuts=256):
    ds_D = downsample(D, rate=downsample_rate, method="triangle")

    filt = -np.ones(n_filt) / (n_filt - 1)
    filt[int((n_filt - 1) / 2)] = 1

    residual = sp.ndimage.convolve1d(ds_D, filt, axis=1)
    mnd, mnt = D.shape
    cuts = []

    for i, _res in enumerate(residual):
        cuts.append([])
        sq_res = np.square(_res)
        med = np.median(sq_res[::4])

        is_bad = (sq_res > 1e2 * med) | np.isnan(sq_res)

        for sub_index in grouper(np.where(is_bad)[0], tol=2):
            s, e = np.min(sub_index) - 1, np.max(sub_index) + 1
            if s < 0 or e > mnt - 1:
                continue
            # sub_res_sum = _res[sub_index].sum()
            cuts[-1].append(
                (
                    downsample_rate * np.min(sub_index) - 1,
                    downsample_rate * np.max(sub_index) + 1,
                )
            )

        if len(cuts[-1]) > max_cuts:
            cuts[-1] = [(0, mnt - 1)]

    return cuts


def apply_cuts(D, cuts, tol=4, method=None):
    fD, T = D.copy(), D.shape[1]
    for i, _cuts in enumerate(cuts):
        for s, e in _cuts:
            if e - s > 1024:
                fD[i, 0] = np.nan
                continue
            elif method == "splice":
                t0, t1 = np.maximum(s - 1, 0), np.minimum(e, T - 1)
                fD[i, t0:t1] = np.linspace(fD[i, t0], fD[i, t1], t1 - t0)
            elif method == "flatten":
                i0, i1, i2, i3 = (
                    np.maximum(s - tol, 0),
                    s,
                    e,
                    np.minimum(e + tol, T - 1),
                )
                if not i0 < i1 < i2 < i3:
                    continue
                d0, d1 = np.median(fD[i, i0:i1]), np.median(fD[i, i2:i3])
                fD[i, i2:] -= d1 - d0
                fD[i, i1:i2] = np.linspace(d0, d0, i2 - i1)
    return fD


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


def bandpass(data, lc, hc, fs, order):
    return sp.signal.filtfilt(
        *sp.signal.butter(order, [2 * lc / fs, 2 * hc / fs], btype="band"),
        data,
        axis=-1,
    )


def lowpass(data, c, fs, order):
    return sp.signal.filtfilt(
        *sp.signal.butter(order, 2 * c / fs, btype="lowpass"), data, axis=-1
    )


def highpass(data, c, fs, order):
    return sp.signal.filtfilt(
        *sp.signal.butter(order, 2 * c / fs, btype="highpass"), data, axis=-1
    )
