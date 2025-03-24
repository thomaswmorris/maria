import numpy as np
import scipy as sp

from ..constants import T_CMB, k_B
from ..functions.radiometry import (
    inverse_planck_spectrum,
    inverse_rayleigh_jeans_spectrum,
    planck_spectrum,
    rayleigh_jeans_spectrum,
)  # noqa


def identity(x: float, **kwargs):
    return x


def rayleigh_jeans_temperature_to_brightness_temperature(T_RJ, nu, **kwargs):
    I_nu = rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu)
    return inverse_planck_spectrum(I_nu=I_nu, nu=nu)


def brightness_temperature_to_rayleigh_jeans_temperature(T_b, nu, **kwargs):
    I_nu = planck_spectrum(T_b=T_b, nu=nu)
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def rayleigh_jeans_temperature_to_radiant_flux(
    T_RJ,
    band,
    spectrum,
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

    return T_RJ * k_B * integral


def radiant_flux_to_rayleigh_jeans_temperature(P, band, spectrum, **kwargs):
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

    return P / (k_B * integral)


def brightness_temperature_to_radiant_flux(T_b, band, spectrum=None, **kwargs):
    test_T_b = np.linspace(np.min(T_b) - 1e-6, np.max(T_b) + 1e-6, 2)

    if spectrum:
        test_T_RJ = inverse_rayleigh_jeans_spectrum(
            planck_spectrum(T_b=test_T_b[:, None], nu=spectrum.side_nu),
            nu=spectrum.side_nu,
        )
        integral = np.trapezoid(
            y=test_T_RJ[:, None, None, None] * np.exp(-spectrum._opacity) * band.passband(spectrum.side_nu),
            x=spectrum.side_nu,
            axis=-1,
        )
        points = (test_T_b, *spectrum.points[:3])
        xi = (
            T_b,
            kwargs["base_temperature"],
            kwargs["zenith_pwv"],
            kwargs["elevation"],
        )
        return k_B * sp.interpolate.interpn(points, integral, xi)

    else:
        test_T_RJ = inverse_rayleigh_jeans_spectrum(planck_spectrum(T_b=test_T_b[:, None], nu=band.nu), nu=band.nu)
        integral = np.trapezoid(y=test_T_RJ * band.passband(band.nu), x=band.nu, axis=-1)
        return k_B * sp.interpolate.interp1d(test_T_b, integral)(T_b)


def radiant_flux_to_brightness_temperature(P, **kwargs):
    raise NotImplementedError()


def dP_dT_CMB(band, spectrum, eps=1e-4, **kwargs):
    P_lo = brightness_temperature_to_radiant_flux(T_CMB - eps / 2, band=band, spectrum=spectrum, **kwargs)
    P_hi = brightness_temperature_to_radiant_flux(T_CMB + eps / 2, band=band, spectrum=spectrum, **kwargs)

    return (P_hi - P_lo) / eps


def cmb_temperature_anisotropy_to_radiant_flux(T_b, band, spectrum, **kwargs):
    return T_b * dP_dT_CMB(band=band, spectrum=spectrum, **kwargs)


def radiant_flux_to_cmb_temperature_anisotropy(P, band, spectrum, **kwargs):
    return P / dP_dT_CMB(band=band, spectrum=spectrum, **kwargs)


def rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel(T_RJ: float, nu: float, pixel_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Kelvin
    nu: frequency, in Hz
    res: resolution, in radians
    """
    return 1e26 * rayleigh_jeans_spectrum(T_RJ=T_RJ, nu=nu) * pixel_area


def spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature(E: float, nu: float, pixel_area: float, **kwargs):
    """
    T_RJ: Rayleigh-Jeans temperature, in Jy/pixel
    nu: frequency, in Hz
    res: resolution, in radians
    """
    I_nu = 1e-26 * E / pixel_area
    return inverse_rayleigh_jeans_spectrum(I_nu=I_nu, nu=nu)


def cmb_temperature_anisotropy_to_radiant_flux_slope(
    band,
    eps: float = 1e-3,
    **kwargs,
):
    test_T_b = T_CMB + np.array([[-eps / 2], [+eps / 2]])
    T_RJ = inverse_rayleigh_jeans_spectrum(
        planck_spectrum(T_b=test_T_b, nu=band.nu),
        nu=band.nu,
    )
    P = k_B * np.trapezoid(T_RJ * band.passband(band.nu), x=band.nu)
    return (P[1] - P[0]) / eps


def cmb_temperature_anisotropy_to_rayleigh_jeans_temperature(
    delta_T: float,
    band,
    **kwargs,
):
    dP_dTCMB = cmb_temperature_anisotropy_to_radiant_flux_slope(band=band)
    return radiant_flux_to_rayleigh_jeans_temperature(
        dP_dTCMB * delta_T,
        band=band,
    )


def rayleigh_jeans_temperature_to_cmb_temperature_anisotropy(
    T_RJ: float,
    band,
    **kwargs,
):
    dP_dTCMB = cmb_temperature_anisotropy_to_radiant_flux_slope(band=band)
    return rayleigh_jeans_temperature_to_radiant_flux(T_RJ, band=band) / dP_dTCMB
