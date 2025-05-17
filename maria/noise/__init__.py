from __future__ import annotations

from functools import partial

import jax
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
