import numpy as np
import scipy as sp


def double_circle_offsets(phase, radius, miss_freq):
    x_c = radius * np.sin(phase)
    y_c = radius * np.cos(phase)

    x_p = radius * np.sin(phase * miss_freq) + x_c
    y_p = radius * np.cos(phase * miss_freq) + y_c
    return x_p, y_p


def double_circle(duration, sample_rate, speed, radius, miss_freq):
    time = np.arange(0, duration, 1 / sample_rate)
    phase = time * speed / radius / miss_freq

    return double_circle_offsets(phase, radius, miss_freq)


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


def daisy(
    duration,
    sample_rate,
    speed=1,
    radius=1,
    petals=10 / np.pi,
    miss_factor=0.15,
    miss_freq=np.sqrt(2),
):
    time = np.arange(0, duration, 1 / sample_rate)
    phase = time * speed / radius

    return daisy_pattern_miss_center(phase, radius, petals, miss_factor, miss_freq)

    # return get_constant_speed_offsets(
    #     daisy_pattern_miss_center, duration, sample_rate, speed, **scan_options
    # )


def back_and_forth(
    duration, sample_rate, speed, x_throw=1, y_throw=0, turnaround_time=1
):
    scan_period = 2 * np.pi * np.sqrt(x_throw**2 + y_throw**2) / speed
    phase = 2 * np.pi * np.arange(0, duration, 1 / sample_rate) / scan_period

    sawtooth = sp.signal.sawtooth(phase, width=0.5)
    smooth_sawtooth = sp.ndimage.gaussian_filter(
        sawtooth, sigma=turnaround_time * sample_rate
    )

    return x_throw * smooth_sawtooth, y_throw * smooth_sawtooth


def stare(duration, sample_rate):
    n_samples = len(np.arange(0, duration, 1 / sample_rate))
    return np.zeros(n_samples), np.zeros(n_samples)
