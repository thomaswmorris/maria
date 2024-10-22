import numpy as np
import scipy as sp  # noqa F401


def compute_angular_fwhm(fwhm_0, z=np.inf, n=1.0, nu=None, l=None):  # noqa F401
    """
    Returns the angular full width at half maximum of a Gaussian beam at distance `z` in
    refractive index `n`. Supply either the wavelength `l` in meters or the frequency `nu` in GHz.

    NOTE: For telescope purposes, `fwhm_0` is the width of the objective/primary.
    """

    if (nu is None) and (l is None):
        raise ValueError("You must supply either a frequency 'f' or wavelength 'l'.")

    w_0 = fwhm_0 / 2

    l = l if l is not None else 2.998e-1 / nu  # noqa F401

    # Rayleigh range
    z_r = np.pi * w_0**2 * n / l

    return 2 * w_0 * np.sqrt(1 / z**2 + 1 / z_r**2)


def compute_physical_fwhm(fwhm_0, z=np.inf, n=1, nu=None, l=None):  # noqa F401
    return z * compute_angular_fwhm(fwhm_0=fwhm_0, z=z, n=n, nu=nu, l=l)


def construct_beam_filter(fwhm, res, beam_profile=None, buffer=1):
    """
    Make a beam filter for an image.
    """

    if beam_profile is None:
        # beam_profile = lambda r, r0: np.where(r <= r0, 1., 0.)

        # a top hat
        def beam_profile(r, r0):
            return np.exp(-((r / r0) ** 16))

    filter_width = buffer * fwhm

    n_side = int(np.maximum(filter_width / res, 3))

    filter_side = np.linspace(-filter_width / 2, filter_width / 2, n_side)
    X, Y = np.meshgrid(filter_side, filter_side, indexing="ij")
    R = np.sqrt(np.square(X) + np.square(Y))
    F = beam_profile(R, fwhm / 2)

    return F / F.sum()


def separably_filter_2d(data, F, tol=1e-2, return_filter=False):
    """
    This is more efficient than 2d convolution
    """

    if F.ndim != 2:
        raise ValueError("F must be two-dimensional.")

    u, s, v = np.linalg.svd(F)
    effective_filter = 0
    filtered_image = 0

    for m in range(len(F)):
        effective_filter += s[m] * u[:, m : m + 1] @ v[m : m + 1]
        filtered_image += s[m] * sp.ndimage.convolve1d(
            sp.ndimage.convolve1d(data, u[:, m], axis=0), v[m], axis=1
        )

        if np.abs(F - effective_filter).mean() < tol:
            break

    return (filtered_image, effective_filter) if return_filter else filtered_image
