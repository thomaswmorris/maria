from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import scipy as sp

logger = logging.getLogger("maria")

VALID_SCAN_KWARGS = [
    "time",
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
    "smoothness",
]


def parse_scan_kwargs(scan_kwargs, default_radius: float = 1.0):
    for kwarg in scan_kwargs:
        if kwarg not in VALID_SCAN_KWARGS:
            raise ValueError(f"Invalid scan kwarg '{kwarg}'")

    size_kwargs = ["radius", "width", "x_throw", "height", "y_throw"]
    if not any([kwarg in scan_kwargs for kwarg in size_kwargs]):
        scan_kwargs["radius"] = default_radius
        logger.warning(
            f"No scan size kwargs (one of {size_kwargs}) were passed. "
            f"Assuming a scan radius of {default_radius:.03e} degrees."
        )

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
        scan_kwargs["speed"] = max(scan_kwargs["x_throw"], scan_kwargs["y_throw"]) / 5

    logger.debug(f"Parsed scan pattern kwargs {scan_kwargs}")

    return scan_kwargs


def lissajous(
    time,
    x_throw,
    y_throw,
    speed,
    freq_ratio=1.193,
    **extra_kwargs,
):  # noqa
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'lissajous'.")

    freq = speed / np.sqrt((x_throw * freq_ratio) ** 2 + y_throw**2)

    x = x_throw * np.cos(freq_ratio * freq * time)
    y = y_throw * np.sin(freq * time)

    return np.stack([x, y])


def double_circle(time, x_throw, y_throw, speed, ratio=0.5, freq_ratio=1.7, **extra_kwargs):
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'double_circle'.")

    radius = x_throw

    a = radius / (1 + 1 / ratio)
    b = a / ratio

    phase = time * speed / np.maximum(a + b * freq_ratio, 1e-16)  # do not divide by zero!

    x = a * np.sin(phase) + b * np.sin(phase * freq_ratio)
    y = a * np.cos(phase) + b * np.cos(phase * freq_ratio)

    return np.stack([x, (y_throw / x_throw) * y])


def daisy_from_phase(phase, a, b, petals, miss_freq):
    x_p = a * np.cos(petals * phase) * np.sin(phase) + b * np.sin(petals * phase) * np.cos(miss_freq * phase)
    y_p = a * np.cos(petals * phase) * np.cos(phase) + b * np.sin(petals * phase) * np.sin(miss_freq * phase)
    X = np.stack([x_p, y_p])
    return (a + b) * X / np.sqrt(np.square(X).sum(axis=0).max())


def daisy(
    time,
    x_throw,
    y_throw,
    speed,
    petals=7 / np.pi,
    miss_factor=0.15,
    miss_freq=np.sqrt(2),
    **extra_kwargs,
):  # noqa
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'daisy'.")

    radius = x_throw

    if radius > 0:
        a = radius / (1 + miss_factor)
        b = a * miss_factor
        max_speed = 0.0
        dp_dt = (speed / radius) if radius > 0.0 else 0.0

        for _ in range(4):
            phase = dp_dt * time
            test_x, test_y = daisy_from_phase(phase, a=a, b=b, petals=petals, miss_freq=miss_freq)
            vx2 = (np.gradient(test_x) / np.gradient(time)) ** 2
            vy2 = (np.gradient(test_y) / np.gradient(time)) ** 2
            max_speed = np.sqrt(vx2 + vy2).max()

            if np.abs(np.log(max_speed / speed)) > 0.01:
                dp_dt *= speed / max_speed
            else:
                break

        x, y = daisy_from_phase(phase, a=a, b=b, petals=petals, miss_freq=miss_freq)

    else:
        x, y = np.zeros((2, len(time)))

    return np.stack([x, (y_throw / x_throw) * y])


def smooth_sawtooth(p, delta=0.01):
    return 1 - 2 * np.arccos((delta - 1) * np.cos(p)) / np.pi


def back_and_forth(t, radius=1, x_throw=None, y_throw=0, speed=1.0, max_accel=np.inf, d=0.01):
    x_throw = x_throw or radius

    factor = 1 / (1 - 2 * np.arccos(1 - d) / np.pi)

    throw = factor * np.sqrt(x_throw**2 + y_throw**2)

    a = np.pi * speed / (2 * throw * (1 - d))
    b = np.sqrt(np.pi * max_accel * np.sqrt(2 * d - d**2) / (2 * throw * (1 - d)))

    dp_dt = np.minimum(a, b)

    x = factor * x_throw * smooth_sawtooth(dp_dt * t, delta=d)
    y = factor * y_throw * smooth_sawtooth(dp_dt * t, delta=d)

    return np.stack([x, y])


def raster(time: float, x_throw: float, y_throw: float, speed: float, n: int = 21, smoothness: float = 2, **extra_kwargs):
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'raster'.")

    sample_rate = 1 / np.median(np.gradient(time))

    raster_distance = np.sqrt((2 * n * x_throw) ** 2 + (2 * y_throw) ** 2)
    raster_duration = raster_distance / speed

    raster_phase = np.linspace(0, 2 * np.pi, 10000)
    raster_x = x_throw * sp.signal.sawtooth(0.5 * n * raster_phase, width=0.5)
    raster_y = y_throw * np.linspace(1, -1, 10000)

    return_distance = np.sqrt((raster_x[-1] + x_throw) ** 2 + (2 * y_throw) ** 2)
    return_duration = return_distance / speed
    period = raster_duration + return_duration

    raster_phase *= raster_duration / period

    phase_samples = [*raster_phase, 2 * np.pi]
    offset_samples = np.stack([[*raster_x, -x_throw], [*raster_y, y_throw]], axis=0)

    phase = 2 * np.pi * (time % period) / period

    offsets = sp.interpolate.interp1d(phase_samples, offset_samples)(phase)

    return sp.ndimage.gaussian_filter1d(offsets, sigma=1, axis=-1)


def stare(time, **extra_kwargs):
    if extra_kwargs:
        logger.warning(f"Ignoring parameters {extra_kwargs} for scan pattern 'stare'.")
    return np.zeros((2, *time.shape))


# def get_constant_speed_offsets(
#     pattern,
#     duration,
#     sample_rate,
#     speed,
#     eps=1e-6,
#     **scan_options,
# ):
#     """This should be cythonized maybe."""

#     speed_per_sample = speed / sample_rate

#     def phase_coroutine():
#         p = 0.0

#         for _ in range(int(duration * sample_rate)):
#             x0, y0 = pattern(p - 0.5 * eps, **scan_options)
#             x1, y1 = pattern(p + 0.5 * eps, **scan_options)

#             ds_dp = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / eps  # speed per phase

#             dp = speed_per_sample / ds_dp
#             p += dp

#             yield p

#     return pattern(np.array([p for p in phase_coroutine()]), **scan_options)


scan_patterns = {
    "stare": {"aliases": [], "generator": stare},
    "daisy": {"aliases": ["daisy_scan"], "generator": daisy},
    "lissajous": {"aliases": ["lissajous_box"], "generator": lissajous},
    "raster": {"aliases": [], "generator": raster},
    "back_and_forth": {"aliases": ["back-and-forth"], "generator": back_and_forth},
    # "grid": {"aliases": [], "generator": grid},
    "double_circle": {"aliases": [], "generator": double_circle},
}

for key in scan_patterns:
    scan_patterns[key]["signature"] = str(inspect.signature(scan_patterns[key]["generator"]))

scan_patterns = pd.DataFrame(scan_patterns).T.sort_index()


def get_scan_pattern_generator(pattern):
    for index, entry in scan_patterns.iterrows():
        if (pattern == index) or (pattern in entry.aliases):
            return entry.generator

    raise ValueError(f"Invalid scan pattern '{pattern}'. Valid scan patterns are {list(scan_patterns.index)}.")


def generate_scan_offsets(time: float, pattern: str, **scan_kwargs):
    f = get_scan_pattern_generator(pattern=pattern)
    return f(time, parse_scan_kwargs(scan_kwargs))
