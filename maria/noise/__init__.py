from __future__ import annotations

from functools import partial

import jax
import numpy as np
from jax import numpy as jnp


@partial(jax.jit, static_argnames=["shape", "sample_rate", "knee", "beta", "seed"])
def generate_noise_with_knee(
    shape: tuple,
    sample_rate: float = 1e0,
    knee: float = 0,
    beta: float = 1.0,
    seed: int = 12345,
):
    """
    Simulate white noise for a given time and NEP.
    """

    noise = jax.random.normal(key=jax.random.key(seed), shape=shape) * jnp.sqrt(sample_rate)

    # pink noise
    if knee > 0:
        f = jnp.fft.fftfreq(n=shape[-1], d=1 / sample_rate)
        a = knee / 2
        pink_noise_power_spectrum = jnp.where(f != 0, a / (jnp.abs(f) ** beta), 0)

        weights = jnp.sqrt(2 * sample_rate * pink_noise_power_spectrum)
        noise += jnp.real(
            jnp.fft.ifft(
                weights * jnp.fft.fft(jax.random.normal(key=jax.random.key(seed), shape=shape)),
            ),
        )

    return noise


def generate_fourier_noise(nx: float = 1024, ny: float = 1024, k0: float = 5e0, beta: float = 8 / 3):
    kx = np.fft.fftfreq(nx, d=1 / nx)
    ky = np.fft.fftfreq(ny, d=1 / ny)
    KY, KX = np.meshgrid(ky, kx)
    P = np.sqrt(k0**2 + KX**2 + KY**2) ** (-beta - 1)
    F = np.fft.fft2(np.sqrt(P) * np.fft.ifft2(np.random.standard_normal(size=(ny, nx)))).real

    return (F - F.mean()) / F.std()
