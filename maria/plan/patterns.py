import numpy as np
import pandas as pd
import scipy as sp


def daisy(
    time,
    radius=1.0,
    speed=None,
    petals=10 / np.pi,
    miss_factor=0.15,
    miss_freq=np.sqrt(2),
):  # noqa
    speed = speed or radius / 5
    phase = time * speed / np.maximum(radius, 1e-6)  # do not divide by zero

    return daisy_pattern_miss_center(
        phase=phase,
        radius=radius,
        petals=petals,
        miss_factor=miss_factor,
        miss_freq=miss_freq,
    )


def daisy_pattern_miss_center(phase, radius, petals, miss_factor, miss_freq):
    shifted_phase = phase + np.pi / 2

    outer_z = np.sin(phase) * np.exp(1j * phase / petals)
    inner_z = (
        miss_factor * np.sin(shifted_phase) * np.exp(1j * miss_freq * phase / petals)
    )

    z = radius * (inner_z + outer_z)

    return np.real(z), np.imag(z)


def grid(time, radius=1, speed=None, n=17, turnaround_time=5):  # noqa
    speed = speed or radius / 5

    duration = np.ptp(time)

    xs = []
    ys = []

    minor_axis = "x"

    timestep = (2 * radius / speed) / (n - 1)

    side = np.linspace(-radius, radius, n)

    while timestep * (len(xs) - n) <= duration:
        minor = []
        major = []

        for i, y in enumerate(side):
            minor.extend(side[:: (-1) ** (i + 1)])
            major.extend(np.repeat(y, n))

        minor.pop(-1), major.pop(-1)

        if minor_axis == "x":
            xs.extend(minor)
            ys.extend(major)
        else:
            xs.extend(major)
            ys.extend(minor)

        minor_axis = "y" if minor_axis == "x" else "x"

    offsets = sp.interpolate.interp1d(
        timestep * np.arange(len(xs)) + time.min(),
        np.c_[xs, ys].T,
        bounds_error=False,
        fill_value="extrapolate",
    )(time)

    return sp.ndimage.gaussian_filter1d(
        offsets, sigma=turnaround_time / timestep, axis=-1
    )


def smooth_sawtooth(phase, width=0.5, smoothness=0.5):
    smooth_phase = sp.ndimage.gaussian_filter1d(
        phase, sigma=smoothness * np.gradient(phase).mean()
    )
    smooth_sawtooth = sp.signal.sawtooth(smooth_phase, width=width)
    return 2 * (smooth_sawtooth - smooth_sawtooth.min()) / smooth_sawtooth.ptp() - 1


def raster(time, radius=1, height=None, speed=0.5, n=16, turnaround_time=0.5):
    width = 2 * radius
    height = height if height is not None else width

    sample_rate = 1 / np.gradient(time).mean()

    start_time = time.min()

    ts, xs, ys = [], [], []

    n_scans = 2 * n + 1
    phase = np.linspace(0, np.pi * n_scans, n_scans * 256)
    raster_period = 2 * n_scans * np.sqrt(width**2 + (height / n_scans) ** 2) / speed

    while start_time < time.max():
        xs.extend(0.5 * width * sp.signal.sawtooth(phase, width=0.5))
        ys.extend(height * np.linspace(0.5, -0.5, len(phase)))
        ts.extend(start_time + np.linspace(0, raster_period, len(phase)))
        start_time = ts[-1] + np.sqrt(width**2 + height**2) / speed

    offsets = sp.interpolate.interp1d(ts, np.c_[xs, ys].T)(time)

    return sp.ndimage.gaussian_filter1d(
        offsets, sigma=turnaround_time * sample_rate, axis=-1
    )


def back_and_forth(time, speed=1, radius=5, turnaround_time=0.5):  # noqa
    return raster(
        time, speed=speed, radius=radius, height=0, turnaround_time=turnaround_time
    )


def stare(time):
    return np.zeros(time.shape), np.zeros(time.shape)


def double_circle(time, speed=None, radius=1.0, ratio=np.pi):
    speed = speed or radius / 10
    phase = time * speed / np.maximum(radius, 1e-6)  # do not divide by zero

    x_p = radius * (np.sin(phase) + np.sin(phase * (1 + ratio))) / 2
    y_p = radius * (np.cos(phase) + np.cos(phase * (1 + ratio))) / 2
    return x_p, y_p


def get_constant_speed_offsets(
    pattern, duration, sample_rate, speed, eps=1e-6, **scan_options
):
    """This should be cythonized maybe."""

    speed_per_sample = speed / sample_rate

    def phase_coroutine():
        p = 0.0

        for _ in range(int(duration * sample_rate)):
            x0, y0 = pattern(p - 0.5 * eps, **scan_options)
            x1, y1 = pattern(p + 0.5 * eps, **scan_options)

            ds_dp = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / eps  # speed per phase

            dp = speed_per_sample / ds_dp
            p += dp

            yield p

    return pattern(np.array([p for p in phase_coroutine()]), **scan_options)


patterns = {
    "stare": {"aliases": [], "generator": stare},
    "daisy": {"aliases": ["daisy_scan"], "generator": daisy},
    "raster": {"aliases": [], "generator": raster},
    "back_and_forth": {"aliases": ["back-and-forth"], "generator": back_and_forth},
    "grid": {"aliases": [], "generator": grid},
    "double_circle": {"aliases": [], "generator": double_circle},
}

for key in patterns:
    patterns[key]["aliases"].append(key)

patterns = pd.DataFrame(patterns).T


def get_pattern_generator(name):
    for index, entry in patterns.iterrows():
        if name in entry.aliases:
            return entry.generator

    raise ValueError(f"Invalid pattern name '{name}'.")
