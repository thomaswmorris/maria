from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def lissajous(
    time,
    radius=1.0,
    speed=None,
    width=None,
    height=None,
    freq_ratio=1.193,
):  # noqa
    width = width or radius / 2
    height = height or width
    speed = speed or width / 10

    freq = speed / np.sqrt((width * freq_ratio) ** 2 + height**2)

    x = width * np.cos(freq_ratio * freq * time)
    y = height * np.sin(freq * time)

    return np.stack([x, y])


def double_circle(time, speed=None, radius=1.0, ratio=0.5, freq_ratio=1.7):
    speed = speed or radius / 10

    a = radius / (1 + 1 / ratio)
    b = a / ratio

    phase = time * speed / np.maximum(a + b * freq_ratio, 1e-16)  # do not divide by zero!

    x_p = a * np.sin(phase) + b * np.sin(phase * freq_ratio)
    y_p = a * np.cos(phase) + b * np.cos(phase * freq_ratio)

    return np.stack([x_p, y_p])


def daisy_from_phase(phase, a, b, petals, miss_freq):
    x_p = a * np.cos(petals * phase) * np.sin(phase) + b * np.sin(petals * phase) * np.cos(miss_freq * phase)
    y_p = a * np.cos(petals * phase) * np.cos(phase) + b * np.sin(petals * phase) * np.sin(miss_freq * phase)
    X = np.stack([x_p, y_p])
    return (a + b) * X / np.sqrt(np.square(X).sum(axis=0).max())


def daisy(
    time,
    radius=1.0,
    speed=None,
    petals=10 / np.pi,
    miss_factor=0.15,
    miss_freq=np.sqrt(2),
):  # noqa
    speed = speed or radius / 10
    a = radius / (1 + miss_factor)
    b = a * miss_factor
    max_speed = 0
    dp_dt = speed / radius

    for iteration in range(4):
        phase = dp_dt * time
        test_x, test_y = daisy_from_phase(phase, a=a, b=b, petals=petals, miss_freq=miss_freq)
        vx2 = (np.gradient(test_x) / np.gradient(time)) ** 2
        vy2 = (np.gradient(test_y) / np.gradient(time)) ** 2
        max_speed = np.sqrt(vx2 + vy2).max()

        if np.abs(np.log(max_speed / speed)) > 0.01:
            dp_dt *= speed / max_speed
        else:
            break

    return daisy_from_phase(phase, a=a, b=b, petals=petals, miss_freq=miss_freq)


def smooth_sawtooth(p, throw=1, delta=0.01):
    return throw * (1 - 2 * np.arccos((1 - delta) * np.sin(p)) / np.pi)


def back_and_forth(t, radius=1, x_throw=None, y_throw=0, speed=1.0, max_accel=np.inf, d=0.01):
    x_throw = x_throw or radius

    factor = 1 / (1 - 2 * np.arccos(1 - d) / np.pi)

    throw = factor * np.sqrt(x_throw**2 + y_throw**2)

    a = np.pi * speed / (2 * throw * (1 - d))
    b = np.sqrt(np.pi * max_accel * np.sqrt(2 * d - d**2) / (2 * throw * (1 - d)))

    dp_dt = np.minimum(a, b)

    x = factor * smooth_sawtooth(dp_dt * t, throw=x_throw, delta=d)
    y = factor * smooth_sawtooth(dp_dt * t, throw=y_throw, delta=d)

    return np.stack([x, y])


# def grid(time, radius=1, speed=None, n=17, turnaround_time=5):  # noqa
#     speed = speed or radius / 5

#     duration = np.ptp(time)

#     xs = []
#     ys = []

#     minor_axis = "x"

#     timestep = (2 * radius / speed) / (n - 1)

#     side = np.linspace(-radius, radius, n)

#     while timestep * (len(xs) - n) <= duration:
#         minor = []
#         major = []

#         for i, y in enumerate(side):
#             minor.extend(side[:: (-1) ** (i + 1)])
#             major.extend(np.repeat(y, n))

#         minor.pop(-1), major.pop(-1)

#         if minor_axis == "x":
#             xs.extend(minor)
#             ys.extend(major)
#         else:
#             xs.extend(major)
#             ys.extend(minor)

#         minor_axis = "y" if minor_axis == "x" else "x"

#     offsets = sp.interpolate.interp1d(
#         timestep * np.arange(len(xs)) + time.min(),
#         np.c_[xs, ys].T,
#         bounds_error=False,
#         fill_value="extrapolate",
#     )(time)

#     return sp.ndimage.gaussian_filter1d(
#         offsets,
#         sigma=turnaround_time / timestep,
#         axis=-1,
#     )


# def smooth_sawtooth(phase, width=0.5, smoothness=0.5):
#     smooth_phase = sp.ndimage.gaussian_filter1d(
#         phase,
#         sigma=smoothness * np.gradient(phase).mean(),
#     )
#     smooth_sawtooth = sp.signal.sawtooth(smooth_phase, width=width)
#     return 2 * (smooth_sawtooth - smooth_sawtooth.min()) / smooth_sawtooth.ptp() - 1


# def raster(time, radius=1, height=None, speed=0.5, n=16, turnaround_time=0.5):
#     width = 2 * radius
#     height = height if height is not None else width

#     sample_rate = 1 / np.gradient(time).mean()

#     start_time = time.min()

#     ts, xs, ys = [], [], []

#     n_scans = 2 * n + 1
#     phase = np.linspace(0, np.pi * n_scans, n_scans * 256)
#     raster_period = 2 * n_scans * np.sqrt(width**2 + (height / n_scans) ** 2) / speed

#     while start_time < time.max():
#         xs.extend(0.5 * width * sp.signal.sawtooth(phase, width=0.5))
#         ys.extend(height * np.linspace(0.5, -0.5, len(phase)))
#         ts.extend(start_time + np.linspace(0, raster_period, len(phase)))
#         start_time = ts[-1] + np.sqrt(width**2 + height**2) / speed

#     offsets = sp.interpolate.interp1d(ts, np.c_[xs, ys].T)(time)

#     return sp.ndimage.gaussian_filter1d(
#         offsets,
#         sigma=turnaround_time * sample_rate,
#         axis=-1,
#     )


# def back_and_forth(time, speed=1, radius=5, turnaround_time=0.5):  # noqa
#     return raster(
#         time,
#         speed=speed,
#         radius=radius,
#         height=0,
#         turnaround_time=turnaround_time,
#     )


def stare(time):
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
    # "raster": {"aliases": [], "generator": raster},
    "back_and_forth": {"aliases": ["back-and-forth"], "generator": back_and_forth},
    # "grid": {"aliases": [], "generator": grid},
    "double_circle": {"aliases": [], "generator": double_circle},
}

for key in scan_patterns:
    scan_patterns[key]["signature"] = str(inspect.signature(scan_patterns[key]["generator"]))

scan_patterns = pd.DataFrame(scan_patterns).T


def get_scan_pattern_generator(name):
    for index, entry in scan_patterns.iterrows():
        if (name == index) or (name in entry.aliases):
            return entry.generator

    raise ValueError(f"Invalid pattern name '{name}'.")
