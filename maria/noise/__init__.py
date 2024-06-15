import dask.array as da
import numpy as np
from tqdm import tqdm

from ..sim.base import BaseSimulation


def generate_noise_with_knee(
    t, n: int = 1, NEP: float = 1e0, knee: float = 0, dask: bool = False
):
    """
    Simulate white noise for a given time and NEP.
    """
    timestep = np.gradient(t, axis=-1).mean()

    if dask:
        noise = da.random.standard_normal(size=(n, len(t))).astype(np.float32)
    else:
        noise = np.random.standard_normal(size=(n, len(t))).astype(np.float32)

    noise *= NEP / np.sqrt(timestep)  # scale the noise

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


class NoiseMixin:
    def _run(self):
        self._simulate_noise()

    def _simulate_noise(self):
        self.data["noise"] = da.from_array(
            np.zeros((self.instrument.n_dets, self.plan.n_time), dtype=np.float32)
        )

        bands = tqdm(
            self.instrument.dets.bands,
            desc="Generating noise",
            disable=not self.verbose,
        )

        for band in bands:
            band_mask = self.instrument.dets.band_name == band.name

            self.data["noise"][band_mask] = generate_noise_with_knee(
                self.plan.time,
                n=band_mask.sum(),
                NEP=band.NEP,
                knee=band.knee,
                dask=True,
            )

    #         # if band.white_noise > 0:
    #         #     self.data["noise"][band_index] += (
    #         #         np.sqrt(self.plan.sample_rate)
    #         #         * np.sqrt(band.white_noise)
    #         #         * np.random.standard_normal(
    #         #             size=(len(band_index), self.plan.n_time)
    #         #         )
    #         #     )

    #         # if band.pink_noise > 0:
    #         #     for i in band_index:
    #         #         self.data["noise"][i] += (
    #         #             np.sqrt(band.pink_noise)
    #         #             * self._spectrum_noise(
    #         #                 spectrum_func=self._pink_spectrum,
    #         #                 size=int(self.plan.n_time),
    #         #                 dt=self.plan.dt,
    #         #             )
    #         #         )

    # def _spectrum_noise(self, spectrum_func, size, dt, amp=2.0):
    #     """
    #     make noise with a certain spectral density
    #     """
    #     freqs = np.fft.rfftfreq(
    #         size, dt
    #     )  # real-fft frequencies (not the negative ones)
    #     spectrum = np.zeros_like(
    #         freqs, dtype="complex"
    #     )  # make complex numbers for spectrum
    #     spectrum[1:] = spectrum_func(
    #         freqs[1:]
    #     )  # get s pectrum amplitude for all frequencies except f=0
    #     phases = np.random.uniform(
    #         0, 2 * np.pi, len(freqs) - 1
    #     )  # random phases for all frequencies except f=0
    #     spectrum[1:] *= np.exp(1j * phases)
    #     noise = np.fft.irfft(spectrum)  # return the reverse fourier transform
    #     return noise

    # def _pink_spectrum(self, f, f_min=0, f_max=np.inf, amp=1.0):
    #     s = (f / amp) ** -0.5
    #     s[np.logical_or(f < f_min, f > f_max)] = 0  # apply band pass
    #     return s


class NoiseSimulation(NoiseMixin, BaseSimulation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(NoiseSimulation, self).__init__(*args, **kwargs)
