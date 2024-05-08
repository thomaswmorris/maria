import numpy as np

from ..sim.base import BaseSimulation


class NoiseMixin:
    def _run(self):
        self._simulate_noise()
        self.tod_data = self.data["noise"]

    def _simulate_noise(self):
        self.data["noise"] = np.zeros((self.instrument.n_dets, self.plan.n_time))

        for band in self.instrument.dets.bands:
            band_index = self.instrument.dets.subset(band_name=band.name).index

            if band.white_noise > 0:
                self.data["noise"][band_index] += (
                    np.sqrt(self.plan.sample_rate)
                    * band.white_noise
                    * np.random.standard_normal(
                        size=(len(band_index), self.plan.n_time)
                    )
                    / band.pW_to_KRJ
                )

            if band.pink_noise > 0:
                for i in band_index:
                    self.data["noise"][i] += (
                        band.pink_noise
                        * self._spectrum_noise(
                            spectrum_func=self._pink_spectrum,
                            size=int(self.plan.n_time),
                            dt=self.plan.dt,
                        )
                        / band.pW_to_KRJ
                    )

    def _spectrum_noise(self, spectrum_func, size, dt, amp=2.0):
        """
        make noise with a certain spectral density
        """
        freqs = np.fft.rfftfreq(
            size, dt
        )  # real-fft frequencies (not the negative ones)
        spectrum = np.zeros_like(
            freqs, dtype="complex"
        )  # make complex numbers for spectrum
        spectrum[1:] = spectrum_func(
            freqs[1:]
        )  # get s pectrum amplitude for all frequencies except f=0
        phases = np.random.uniform(
            0, 2 * np.pi, len(freqs) - 1
        )  # random phases for all frequencies except f=0
        spectrum[1:] *= np.exp(1j * phases)
        noise = np.fft.irfft(spectrum)  # return the reverse fourier transform
        return noise

    def _pink_spectrum(self, f, f_min=0, f_max=np.inf, amp=1.0):
        s = (f / amp) ** -0.5
        s[np.logical_or(f < f_min, f > f_max)] = 0  # apply band pass
        return s


class NoiseSimulation(NoiseMixin, BaseSimulation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(NoiseSimulation, self).__init__(*args, **kwargs)
