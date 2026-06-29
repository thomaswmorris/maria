from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import scipy as sp

from ..utils import get_rotation_matrix_2d

logger = logging.getLogger("maria")

VALID_SCAN_KWARGS = [
    "t",
    "radius",
    "width",
    "height",
    "x_throw",
    "y_throw",
    "speed",
    "n",
    "petals",
    "ratio",
    "freq_ratio",
    "miss_factor",
    "miss_freq",
    "rotation_period",
    "smoothness",
]


def parse_scan_kwargs(scan_kwargs, default_radius: float = 1.0):
    for kwarg in scan_kwargs:
        if kwarg not in VALID_SCAN_KWARGS:
            raise ValueError(f"Invalid scan kwarg '{kwarg}'")

    size_kwargs = ["radius", "width", "x_throw", "height", "y_throw"]
    if not any([kwarg in scan_kwargs for kwarg in size_kwargs]):
        if default_radius is None:
            default_radius = 1.0
            logger.warning(
                f"No scan size kwargs (one of {size_kwargs}) were passed. "
                f"Assuming a scan radius of {default_radius:.03e} degrees."
            )
        scan_kwargs["radius"] = default_radius

    if "x_throw" not in scan_kwargs:
        if "radius" in scan_kwargs:
            scan_kwargs["x_throw"] = scan_kwargs.pop("radius")
        elif "width" in scan_kwargs:
            scan_kwargs["x_throw"] = 0.5 * scan_kwargs.pop("width")
        elif "y_throw" in scan_kwargs:
            scan_kwargs["x_throw"] = scan_kwargs["y_throw"]
        else:
            scan_kwargs["x_throw"] = 0.5 * scan_kwargs.pop("height")

    if "y_throw" not in scan_kwargs:
        if "height" in scan_kwargs:
            scan_kwargs["y_throw"] = 0.5 * scan_kwargs.pop("height")
        else:
            scan_kwargs["y_throw"] = scan_kwargs["x_throw"]

    if "speed" not in scan_kwargs:
        scan_kwargs["speed"] = max(scan_kwargs["x_throw"], scan_kwargs["y_throw"]) / 4

    logger.debug(f"Parsed scan pattern kwargs {scan_kwargs}")

    return scan_kwargs


def lissajous(
    t,
    x_throw,
    y_throw,
    speed,
    freq_ratio=1.193,
    **extra_kwargs,
):  # noqa
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'lissajous'.")

    if x_throw == 0 and y_throw == 0:
        return np.zeros((len(t), 2))

    freq = speed / np.sqrt((x_throw * freq_ratio) ** 2 + y_throw**2)

    x = x_throw * np.cos(freq_ratio * freq * t)
    y = y_throw * np.sin(freq * t)

    return np.stack([x, y], axis=-1)


def double_circle(t, x_throw, y_throw, speed, ratio=0.5, freq_ratio=1.7, **extra_kwargs):
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'double_circle'.")

    if x_throw == 0 and y_throw == 0:
        return np.zeros((len(t), 2))

    radius = max(x_throw, y_throw)

    a = radius / (1 + 1 / ratio)
    b = a / ratio

    phase = t * speed / np.maximum(a + b * freq_ratio, 1e-16)  # do not divide by zero!

    x = a * np.sin(phase) + b * np.sin(phase * freq_ratio)
    y = a * np.cos(phase) + b * np.cos(phase * freq_ratio)

    return np.stack([(x_throw / radius) * x, (y_throw / radius) * y], axis=-1)


def daisy_from_phase(phase, a, b, petals, miss_freq):
    x_p = a * np.cos(petals * phase) * np.sin(phase) + b * np.sin(petals * phase) * np.cos(miss_freq * phase)
    y_p = a * np.cos(petals * phase) * np.cos(phase) + b * np.sin(petals * phase) * np.sin(miss_freq * phase)
    X = np.stack([x_p, y_p], axis=-1)
    return (a + b) * X / np.sqrt(np.square(X).sum(axis=-1).max())


def daisy(
    t,
    x_throw,
    y_throw,
    speed,
    petals=np.sqrt(np.e),
    miss_factor=0.2,
    miss_freq=0.1,
    **extra_kwargs,
):  # noqa
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'daisy'.")

    if x_throw == 0 and y_throw == 0:
        return np.zeros((len(t), 2))

    radius = max(x_throw, y_throw)

    a = radius / (1 + miss_factor)
    b = a * miss_factor
    max_speed = 0.0
    dp_dt = (speed / radius) if radius > 0.0 else 0.0
    dp = dp_dt * np.gradient(t)
    # dp *= (1 + 0.5 * np.cos(t / (100 * np.pi)))

    for _ in range(4):
        phase = np.cumsum(dp)
        test_x, test_y = daisy_from_phase(phase, a=a, b=b, petals=petals, miss_freq=miss_freq).T
        vx2 = (np.gradient(test_x) / np.gradient(t)) ** 2
        vy2 = (np.gradient(test_y) / np.gradient(t)) ** 2
        max_speed = np.sqrt(vx2 + vy2).max()

        if np.abs(np.log(max_speed / speed)) > 0.01:
            dp *= speed / max_speed
        else:
            break

    x, y = daisy_from_phase(phase, a=a, b=b, petals=petals, miss_freq=miss_freq).T
    return np.stack([(x_throw / radius) * x, (y_throw / radius) * y], axis=-1)


def smooth_sawtooth(p, delta=0.01):
    norm = 1 / (2 * np.arccos(delta - 1) / np.pi - 1)
    return norm * (1 - 2 * np.arccos((delta - 1) * np.cos(p)) / np.pi)


def back_and_forth(t, x_throw=1, y_throw=0, speed=1.0, max_accel=np.inf, d=0.01):

    factor = 1 / (1 - 2 * np.arccos(1 - d) / np.pi)

    if x_throw == 0 and y_throw == 0:
        return np.zeros((len(t), 2))

    throw = factor * np.sqrt(x_throw**2 + y_throw**2)

    a = np.pi * speed / (2 * throw * (1 - d))
    b = np.sqrt(np.pi * max_accel * np.sqrt(2 * d - d**2) / (2 * throw * (1 - d)))
    dp_dt = np.minimum(a, b)

    x = factor * x_throw * smooth_sawtooth(dp_dt * t, delta=d)
    y = factor * y_throw * smooth_sawtooth(dp_dt * t, delta=d)

    return np.stack([x, y], axis=-1)


def raster(
    t: float,
    x_throw: float,
    y_throw: float,
    speed: float,
    n: int = [(11, 1), (1, 11)],
    d: float = 1e-1,
    rotation_period: float = np.inf,
    samples_per_period: int = 10000,
    **extra_kwargs,
):

    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'raster'.")

    total_duration = 0.0

    if x_throw == 0 and y_throw == 0:
        return np.zeros((len(t), 2))

    period = 0
    period_times_list = []
    period_offsets_list = []

    current_direction = np.array([1, -1])

    while total_duration < np.ptp(t):
        nx, ny = n[period % len(n)]

        period_phase = np.linspace(0, np.pi, samples_per_period)
        period_offsets = np.stack(
            [x_throw * smooth_sawtooth(nx * period_phase, delta=d), y_throw * smooth_sawtooth(ny * period_phase, delta=d)],
            axis=-1,
        )

        max_step = np.sqrt(np.sum(np.diff(period_offsets, axis=0) ** 2, axis=-1)).max()

        # period_distance = np.sum(np.sqrt(np.sum(np.diff(period_offsets, axis=0) ** 2, axis=-1)))
        period_duration = max_step * samples_per_period / speed

        period_times_list.append(total_duration + np.linspace(0, period_duration, samples_per_period)[:-1])
        period_offsets_list.append(current_direction * period_offsets[:-1])
        total_duration += period_duration
        current_direction = -np.sign(period_offsets_list[-1][-1])

        period += 1

        logger.debug(f"Raster period {period} (n = {nx},{ny})")

    time_samples = np.concatenate(period_times_list)
    offsets_samples = np.concatenate(period_offsets_list)

    offsets = sp.interpolate.interp1d(time_samples, offsets_samples, axis=0, kind="linear")(t - t.min())

    if np.isfinite(rotation_period):
        rotation_phase = (2 * np.pi * (t - t[0]) / rotation_period) % (2 * np.pi)
        offsets = np.einsum("ti,tij->tj", offsets, get_rotation_matrix_2d(rotation_phase))

    return offsets


def stare(t, **extra_kwargs):
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'stare'.")
    return np.zeros((*t.shape, 2))


# def get_constant_speed_offsets(
#     pattern,
#     duration,
#     sample_rate,
#     speed,
#     eps=1e-6,
#     **scan_parameters,
# ):
#     """This should be cythonized maybe."""

#     speed_per_sample = speed / sample_rate

#     def phase_coroutine():
#         p = 0.0

#         for _ in range(int(duration * sample_rate)):
#             x0, y0 = pattern(p - 0.5 * eps, **scan_parameters)
#             x1, y1 = pattern(p + 0.5 * eps, **scan_parameters)

#             ds_dp = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / eps  # speed per phase

#             dp = speed_per_sample / ds_dp
#             p += dp

#             yield p

#     return pattern(np.array([p for p in phase_coroutine()]), **scan_parameters)


scan_types = {
    "stare": {"aliases": [], "generator": stare},
    "daisy": {"aliases": ["daisy_scan"], "generator": daisy},
    "lissajous": {"aliases": ["lissajous_box"], "generator": lissajous},
    "raster": {"aliases": [], "generator": raster},
    "back_and_forth": {"aliases": ["back-and-forth"], "generator": back_and_forth},
    # "grid": {"aliases": [], "generator": grid},
    "double_circle": {"aliases": [], "generator": double_circle},
}

for key in scan_types:
    scan_types[key]["signature"] = str(inspect.signature(scan_types[key]["generator"]))

scan_types = pd.DataFrame(scan_types).T.sort_index()


def get_scan_type_generator(pattern):
    for index, entry in scan_types.iterrows():
        if (pattern == index) or (pattern in entry.aliases):
            return entry.generator

    raise ValueError(f"Invalid scan pattern '{pattern}'. Valid scan patterns are {list(scan_types.index)}.")


def generate_scan_offsets(t: float, pattern: str, **scan_kwargs):
    f = get_scan_type_generator(pattern=pattern)
    return f(t, parse_scan_kwargs(scan_kwargs))
