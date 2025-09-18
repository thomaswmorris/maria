import numpy as np
import scipy as sp

from ..constants import T_CMB, k_B
from ..errors import ShapeError
from ..functions.radiometry import (
    inverse_planck_spectrum,
    inverse_rayleigh_jeans_spectrum,
    planck_spectrum,
    rayleigh_jeans_spectrum,
)  # noqa


def identity(x: float, **kwargs):
    return x


def cmb_temperature_anisotropy_to_brightness_temperature(dT_CMB, **kwargs):
    return dT_CMB + T_CMB


def brightness_temperature_to_cmb_temperature_anisotropy(T_b, **kwargs):
    return T_b - T_CMB


def rayleigh_jeans_temperature_to_brightness_temperature(T_RJ, nu, **kwargs):
    I_nu = rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu)
    return inverse_planck_spectrum(I_nu=I_nu, nu=nu)


def brightness_temperature_to_rayleigh_jeans_temperature(T_b, nu, **kwargs):
    I_nu = planck_spectrum(T_b=T_b, nu=nu)
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def rayleigh_jeans_temperature_to_radiant_flux(
    T_RJ,
    band,
    polarized: bool = False,
    spectrum=None,
    **kwargs,
):
    """
    nu: frequency, in Hz
    band: response to a Rayleigh-Jeans source
    """
    if spectrum:
        integral = band.compute_nu_integral(
            spectrum=spectrum,
            zenith_pwv=kwargs["zenith_pwv"],
            base_temperature=kwargs["base_temperature"],
            elevation=kwargs["elevation"],
        )

    else:
        integral = band.compute_nu_integral(spectrum=None)

    return (0.5 if polarized else 1.0) * k_B * integral * T_RJ


def radiant_flux_to_rayleigh_jeans_temperature(P, band, polarized: bool = False, spectrum=None, **kwargs):
    """
    nu: frequency, in Hz
    passband: response to a Rayleigh-Jeans source
    """

    if spectrum:
        integral = band.compute_nu_integral(
            spectrum=spectrum,
            zenith_pwv=kwargs["zenith_pwv"],
            base_temperature=kwargs["base_temperature"],
            elevation=kwargs["elevation"],
        )

    else:
        integral = band.compute_nu_integral(spectrum=None)

    return P / ((0.5 if polarized else 1.0) * k_B * integral)


def brightness_temperature_to_radiant_flux_explicit(
    T_b, band, polarized: bool = False, spectrum=None, eps: float = 1e-4, **kwargs
):
    if T_b.ndim > 1:
        raise ShapeError("'T_b' must be one-dimensional")

    # stokes_factor = 0.5 if polarized else 1.0
    shaped_T_b = np.expand_dims(T_b, axis=0)

    if spectrum:
        # add dimensions to broadcast with the atmospheric spectrum
        shaped_T_b = np.expand_dims(shaped_T_b, axis=(0, 1, 2))
        shaped_nu = np.expand_dims(spectrum.side_nu, -1)
        sample_T_RJ = inverse_rayleigh_jeans_spectrum(
            planck_spectrum(T_b=shaped_T_b, nu=shaped_nu),
            nu=shaped_nu,
        )

        # this has shape (spectrum.shape, T_b.size)
        integral_samples = np.trapezoid(
            y=sample_T_RJ * (np.exp(-np.expand_dims(spectrum._opacity, -1)) * band.passband(shaped_nu)),
            x=spectrum.side_nu,
            axis=-2,
        )

        points = spectrum.points[:3]
        xi = (
            kwargs["base_temperature"],
            kwargs["zenith_pwv"],
            kwargs["elevation"],
        )
        integral = sp.interpolate.RegularGridInterpolator(points, integral_samples)(xi)

    else:
        shaped_nu = np.expand_dims(band.nu.Hz, -1)
        sample_T_RJ = inverse_rayleigh_jeans_spectrum(
            planck_spectrum(T_b=shaped_T_b, nu=shaped_nu),
            nu=shaped_nu,
        )
        integral = np.trapezoid(y=sample_T_RJ * band.passband(shaped_nu), x=band.nu.Hz, axis=-2)

    return (0.5 if polarized else 1.0) * k_B * integral


def brightness_temperature_to_radiant_flux(T_b, band, polarized: bool = False, spectrum=None, eps: float = 1e-4, **kwargs):
    T_b_lo = np.min(T_b) - eps / 2
    T_b_hi = np.min(T_b) + eps / 2

    # this has shape (*kwargs.shape, 2)
    P = brightness_temperature_to_radiant_flux_explicit(
        T_b=np.array([T_b_lo, T_b_hi]), band=band, polarized=polarized, spectrum=spectrum, **kwargs
    )

    t = (T_b - T_b_lo) / eps
    return t * P[..., 1] + (1 - t) * P[..., 0]


def radiant_flux_to_brightness_temperature(P, **kwargs):
    raise NotImplementedError()


def rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel(T_RJ: float, nu: float, pixel_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Kelvin
    nu: frequency, in Hz
    res: resolution, in radians
    """
    return rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu) * pixel_area


def spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature(E: float, nu: float, pixel_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Jy/pixel
    nu: frequency, in Hz
    res: resolution, in radians
    """
    I_nu = E / pixel_area
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def dP_dT_CMB(band, polarized=False, spectrum=None, eps=1e-4, **kwargs):
    sample_T_b = np.array([T_CMB - eps / 2, T_CMB + eps / 2])
    P = brightness_temperature_to_radiant_flux_explicit(
        T_b=sample_T_b, band=band, polarized=polarized, spectrum=spectrum, **kwargs
    )
    return (P[..., 1] - P[..., 0]) / eps


def cmb_temperature_anisotropy_to_radiant_flux(T_b, band, polarized=False, spectrum=None, **kwargs):
    return T_b * dP_dT_CMB(band=band, polarized=polarized, spectrum=spectrum, **kwargs)


def radiant_flux_to_cmb_temperature_anisotropy(P, band, polarized=False, spectrum=None, **kwargs):
    return P / dP_dT_CMB(band=band, polarized=polarized, spectrum=spectrum, **kwargs)


def T_RJ_per_T_CMB(
    band,
    eps: float = 1e-3,
    **kwargs,
):
    """
    Color correction for NO ATMOSPHERE
    """
    test_T_b = T_CMB + np.array([[-eps / 2], [+eps / 2]])
    T_RJ = inverse_rayleigh_jeans_spectrum(
        planck_spectrum(T_b=test_T_b, nu=band.nu.Hz),
        nu=band.nu.Hz,
    )
    P = k_B * np.trapezoid(T_RJ * band.passband(band.nu.Hz), x=band.nu.Hz, axis=-1)
    return radiant_flux_to_rayleigh_jeans_temperature((P[1] - P[0]) / eps, spectrum=None, band=band)


def cmb_temperature_anisotropy_to_rayleigh_jeans_temperature(
    delta_T_CMB: float,
    band,
    **kwargs,
):
    return delta_T_CMB * T_RJ_per_T_CMB(band=band)


def rayleigh_jeans_temperature_to_cmb_temperature_anisotropy(
    T_RJ: float,
    band,
    **kwargs,
):
    return T_RJ / T_RJ_per_T_CMB(band=band)


def spectral_radiance_to_spectral_flux_density_per_pixel(E: float, pixel_area: float, **kwargs):
    """
    E: Spectral flux density per area, in Jy/sr
    pixel_area: pixel area, in steradians
    """
    return E * pixel_area


def spectral_flux_density_per_pixel_to_spectral_radiance(E: float, pixel_area: float, **kwargs):
    """
    E: Spectral flux density per pixel, in Jy/pixel
    pixel_area: pixel area, in steradians
    """
    return E / pixel_area


def spectral_flux_density_per_beam_to_spectral_flux_density_per_pixel(
    E: float, pixel_area: float, beam_area: float, **kwargs
):
    """
    E: Spectral flux density per area, in Jy/sr
    pixel_area: pixel area, in steradians
    """
    return E * pixel_area / beam_area


def spectral_flux_density_per_pixel_to_spectral_flux_density_per_beam(
    E: float, pixel_area: float, beam_area: float, **kwargs
):
    """
    E: Spectral flux density per pixel, in Jy/pixel
    pixel_area: pixel area, in steradians
    """
    return E * beam_area / pixel_area
