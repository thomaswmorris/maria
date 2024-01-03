import numpy as np
import scipy as sp  # noqa F401


def angular_fwhm(fwhm_0, z=np.inf, n=1, f=None, l=None):  # noqa F401
    """
    Returns the angular full width at half maximum of a Gaussian beam with waist `w_0` at distance `z` in
    refractive index `n`. Supply either the wavelength `l` in meters or the frequency `f` in Hz.

    NOTE: The waist of a beam is half the diameter at its smallest point.
    For telescope purposes, this will be half the width of the objective/primary.
    """

    w_0 = fwhm_0 / 2

    l = l if l is not None else 2.998e8 / f  # noqa F401

    # Rayleigh range
    z_r = np.pi * w_0**2 * n / l

    return 2 * w_0 * np.sqrt(1 / z**2 + 1 / z_r**2)


# def gaussian_beam_angular_fwhm(z, w_0, n=1, f=None, l=None):  # noqa F401
#     """
#     Returns the angular full width at half maximum of a Gaussian beam with waist `w_0` at distance `z` in
#     refractive index `n`. Supply either the wavelength `l` in meters or the frequency `f` in GHz.

#     NOTE: The waist of a beam is half the diameter at its smallest point.
#     For telescope purposes, this will be half the width of the objective/primary.
#     """
#     l = l if l is not None else 2.998e8 / (1e9 * f)  # noqa F401

#     # Rayleigh range
#     z_r = np.pi * w_0**2 * n / l

#     return np.sqrt(2 * np.log(2)) * w_0 * np.sqrt(1 / z**2 + 1 / z_r**2)


# def gaussian_beam_physical_fwhm(z, w_0, n=1, f=None, l=None):  # noqa F401
#     return z * gaussian_beam_angular_fwhm(z=z, w_0=w_0, n=n, f=f, l=l)  # noqa F401


def make_beam_filter(fwhm, res, beam_profile=None, buffer=1):
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


def separably_filter(data, F, tol=1e-2, return_filter=False):
    """
    This is more efficient than 2d convolution
    """

    assert data.ndim == 2

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
