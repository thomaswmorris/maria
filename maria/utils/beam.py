import numpy as np
import scipy as sp


def gaussian_beam_angular_fwhm(z, w_0, n=1, f=None, l=None):  # noqa F401
    """
    Returns the angular full width at half maximum of a Gaussian beam with waist `w_0` at distance `z` in
    refractive index `n`. Supply either the wavelength `l` in meters or the frequency `f` in GHz.

    NOTE: The waist of a beam is half the diameter at its smallest point.
    For telescope purposes, this will be half the width of the objective/primary.
    """
    l = l if l is not None else 2.998e8 / (1e9 * f)  # noqa F401

    # Rayleigh range
    z_r = np.pi * w_0**2 * n / l

    return np.sqrt(2 * np.log(2)) * w_0 * np.sqrt(1 / z**2 + 1 / z_r**2)


def gaussian_beam_physical_fwhm(z, w_0, n=1, f=None, l=None):  # noqa F401
    return z * gaussian_beam_angular_fwhm(z=z, w_0=w_0, n=n, f=f, l=l)  # noqa F401


def make_beam_filter(waist, res, func, width_per_waist=1.2):
    filter_width = width_per_waist * waist
    n_filter = 2 * int(np.ceil(0.5 * filter_width / res)) + 1

    filter_side = 0.5 * np.linspace(-filter_width, filter_width, n_filter)

    FILTER_X, FILTER_Y = np.meshgrid(filter_side, filter_side, indexing="ij")
    FILTER_R = np.sqrt(np.square(FILTER_X) + np.square(FILTER_Y))

    FILTER = func(FILTER_R, 0.5 * waist)
    FILTER /= FILTER.sum()

    return FILTER


def separate_beam_filter(F, tol=1e-2):
    u, s, v = np.linalg.svd(F)
    eff_filter = 0
    for m, (_u, _s, _v) in enumerate(zip(u.T, s, v)):
        eff_filter += _s * np.outer(_u, _v)
        if np.abs(F - eff_filter).sum() < tol:
            break

    return u.T[: m + 1], s[: m + 1], v[: m + 1]


def separably_filter(M, F, tol=1e-2):
    u, s, v = separate_beam_filter(F, tol=tol)

    filt_M = 0
    for _u, _s, _v in zip(u, s, v):
        filt_M += _s * sp.ndimage.convolve1d(
            sp.ndimage.convolve1d(M.astype(float), _u, axis=0), _v, axis=1
        )

    return filt_M
