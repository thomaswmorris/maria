import numpy as np


def get_kernel(n, kind='triangle'):
    if kind == 'triangle':
        kernel = np.r_[np.linspace(0, 1, n + 1)[1:], np.linspace(1, 0, n + 1)[1:-1]]

        return kernel / kernel.sum()


def downsample(DATA, rate, axis=-1, method='triangle'):
    if method == 'flat':
        _DATA = np.swapaxes(DATA, 0, axis)
        cs_data = np.cumsum(_DATA, axis=0)
        return np.swapaxes((cs_data[rate::rate] - cs_data[:-rate:rate]) / rate, 0, axis)

    else:
        if rate == 1:
            return DATA
        if rate < 1:
            raise ValueError('downsample rate must be an integer >= 1')

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


def decompose(DATA, mode='uv', downsample_rate=1):
    downsampled_data = downsample(DATA, rate=downsample_rate, method='triangle')
    rms = downsampled_data.std(axis=-1)
    u, s, v = np.linalg.svd(downsampled_data / rms[:, None], full_matrices=False)
    uv_norm = v.std(axis=-1) * np.sign(u.mean(axis=0))
    s_norm = np.sqrt(np.square(s).sum())
    u *= s_norm * rms[:, None] * uv_norm[None, :]
    s /= s_norm

    if mode == 'uv':
        return np.matmul(u, np.diag(s)), np.matmul(
            np.linalg.pinv(np.matmul(u, np.diag(s))), DATA
        )
    elif mode == 'usv':
        return u, s, np.matmul(np.linalg.pinv(np.matmul(u, np.diag(s))), DATA)
    else:
        raise ValueError()
