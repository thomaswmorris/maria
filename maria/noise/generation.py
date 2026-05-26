from __future__ import annotations

import jax
import numpy as np

from ..utils import compute_diameter, generate_spatial_basis

DEFAULT_NOISE_SIM_KWARGS = {"corr_prop": 0.5, "spatial_scale": 0.9}


def generate_noise_with_knee(
    shape: tuple,
    sample_rate: float = 1.0,
    knee: float = 0.0,
    beta: float = 1.0,
    basis: float = None,
    corr_prop: float = 0.0,
    seed: int = 12345,
):
    """
    Simulate white noise for a given time and NEP.
    """

    noise = np.sqrt(sample_rate) * np.random.standard_normal(shape)

    # pink noise
    if knee > 0:
        f = np.fft.fftfreq(n=shape[-1], d=1 / sample_rate)
        a = knee / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            pink_noise_power_spectrum = np.where(f != 0, a / (np.abs(f) ** beta), 0)

        weights = np.sqrt(2 * sample_rate * pink_noise_power_spectrum)
        pink_noise = np.real(
            np.fft.ifft(
                weights * np.fft.fft(jax.random.normal(key=jax.random.key(seed), shape=shape)),
            ),
        )

        if basis is not None:
            noise_modes = generate_noise_with_knee(
                shape=(basis.shape[-1], shape[-1]),
                sample_rate=sample_rate,
                knee=knee,
            )

            pink_noise = np.sqrt(corr_prop) * basis @ noise_modes + np.sqrt(1 - corr_prop) * pink_noise

        noise += pink_noise

    return noise


def generate_fourier_noise(nx: float = 1024, ny: float = 1024, k0: float = 5e0, beta: float = 8 / 3):
    kx = np.fft.fftfreq(nx, d=1 / nx)
    ky = np.fft.fftfreq(ny, d=1 / ny)
    KY, KX = np.meshgrid(ky, kx)
    P = np.sqrt(k0**2 + KX**2 + KY**2) ** (-beta - 1)
    F = np.fft.fft2(np.sqrt(P) * np.fft.ifft2(np.random.standard_normal(size=(ny, nx)))).real

    return (F - F.mean()) / F.std()
