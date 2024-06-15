import numpy as np


def generate_noise_with_knee(t, n: int = 1, NEP: float = 1e0, knee: float = 0):
    """
    Simulate white noise for a given time and NEP.
    """
    timestep = np.gradient(t).mean()
    noise = NEP / np.sqrt(timestep) * np.random.standard_normal(size=(n, len(t)))

    if knee > 0:
        f = np.fft.fftfreq(len(t), d=timestep)
        a = knee * NEP**2 / 2
        with np.errstate(divide="ignore"):
            pink_noise_power_spectrum = np.where(f != 0, a / np.abs(f), 0)

        weights = np.sqrt(2 * pink_noise_power_spectrum / timestep)
        noise += np.real(
            np.fft.ifft(
                weights * np.fft.fft(np.random.standard_normal(size=(n, len(t))))
            )
        )

    # pink noise
    return noise
