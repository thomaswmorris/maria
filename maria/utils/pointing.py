import numpy as np


def daisy_pattern(p, petals, radius):
    distance_from_center = radius * np.sin(p)
    x = distance_from_center * np.cos(p / petals)
    y = distance_from_center * np.sin(p / petals)
    return x, y


def get_constant_speed_offsets(
    pattern, integration_time, sample_rate, speed, eps=1e-6, **scan_options
):
    """This should be cythonized maybe."""

    speed_per_sample = speed / sample_rate

    def phase_coroutine():
        p = 0.0

        for _ in range(int(integration_time * sample_rate)):
            x0, y0 = pattern(p - 0.5 * eps, **scan_options)
            x1, y1 = pattern(p + 0.5 * eps, **scan_options)

            ds_dp = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / eps  # speed per phase

            dp = speed_per_sample / ds_dp
            p += dp

            yield p

    return pattern(np.array([p for p in phase_coroutine()]), **scan_options)


def get_daisy_offsets_constant_speed(
    integration_time, sample_rate, speed, **scan_options
):
    return get_constant_speed_offsets(
        daisy_pattern, integration_time, sample_rate, speed, **scan_options
    )


def get_back_and_forth_offsets(
    integration_time, sample_rate, speed, x_throw, y_throw, taper=0.1
):
    scan_period = np.sqrt(x_throw**2 + y_throw**2) / speed
    phase = 2 * np.pi * np.arange(0, integration_time, 1 / sample_rate) / scan_period
    smooth_sawtooth = 1 - 2 * np.arccos((1 - taper) * np.sin(phase)) / np.pi

    return x_throw * smooth_sawtooth, y_throw * smooth_sawtooth


def get_stare_offsets(integration_time, sample_rate):
    n_samples = len(np.arange(0, integration_time, 1 / sample_rate))
    return np.zeros(n_samples), np.zeros(n_samples)


def get_offsets(scan_pattern, integration_time, sample_rate, **scan_options):
    """
    Returns x and y offsets
    """

    if scan_pattern == "stare":
        return get_stare_offsets(
            integration_time=integration_time, sample_rate=sample_rate, **scan_options
        )

    if scan_pattern == "daisy":
        return get_daisy_offsets_constant_speed(
            integration_time=integration_time, sample_rate=sample_rate, **scan_options
        )

    elif scan_pattern == "back_and_forth":
        return get_back_and_forth_offsets(
            integration_time=integration_time, sample_rate=sample_rate, **scan_options
        )

    else:
        raise ValueError(f"'{scan_pattern}' is not a valid scan pattern.")
