import numpy as np
import scipy as sp


def daisy(
    time,
    radius=1,
    speed=None,
    petals=10 / np.pi,
    miss_factor=0.15,
    miss_freq=np.sqrt(2),
):  # noqa
    speed = speed or radius / 5
    phase = time * speed / radius

    return daisy_pattern_miss_center(phase, radius, petals, miss_factor, miss_freq)


def double_circle(time, radius=1, speed=None, miss_freq=np.sqrt(2)):  # noqa
    speed = speed or radius / 5
    phase = time * speed / radius / miss_freq

    return double_circle_offsets(phase, radius, miss_freq)


def grid(time, radius=1, speed=None, n=17, turnaround_time=5):  # noqa
    speed = speed or radius / 5

    duration = time.ptp()

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


def back_and_forth(time, speed, x_throw=1, y_throw=0, turnaround_time=5):  # noqa
    sample_rate = 1 / np.gradient(time).mean()
    scan_period = 2 * np.pi * np.sqrt(x_throw**2 + y_throw**2) / speed
    phase = 2 * np.pi * time / scan_period

    sawtooth = sp.signal.sawtooth(phase, width=0.5)
    smooth_sawtooth = sp.ndimage.gaussian_filter(
        sawtooth, sigma=turnaround_time * sample_rate
    )  # noqa

    return x_throw * smooth_sawtooth, y_throw * smooth_sawtooth


def stare(time):
    return np.zeros(time.shape), np.zeros(time.shape)


def double_circle_offsets(phase, radius, miss_freq):
    x_c = radius * np.sin(phase)
    y_c = radius * np.cos(phase)

    x_p = radius * np.sin(phase * miss_freq) + x_c
    y_p = radius * np.cos(phase * miss_freq) + y_c
    return x_p, y_p


def daisy_pattern_miss_center(phase, radius, petals, miss_factor, miss_freq):
    shifted_phase = phase + np.pi / 2

    outer_z = np.sin(phase) * np.exp(1j * phase / petals)
    inner_z = (
        miss_factor * np.sin(shifted_phase) * np.exp(1j * miss_freq * phase / petals)
    )

    z = radius * (inner_z + outer_z)

    # assert False

    return np.real(z), np.imag(z)


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
