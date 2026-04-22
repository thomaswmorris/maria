import numpy as np


def compute_resolution_precision(x):
    if np.size(x) > 1:
        dx = np.gradient(np.sort([0, *np.ravel(x)]))
        min_dx = np.nanmin(np.where(dx > 0, dx, np.nan))
        if min_dx > 0:
            return max(4, int(-np.floor(np.log10(min_dx))) + 1)
    return 4


def round_sig_figs(x, sig_figs):
    power = np.floor(np.log10(x))
    return np.round(np.round(x * 10**-power, sig_figs - 1) * 10**power, 10)
