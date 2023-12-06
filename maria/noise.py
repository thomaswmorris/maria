import numpy as np

from . import base


class WhiteNoiseSimulation(base.BaseSimulation):
    """
    White noise! It's Gaussian.
    """

    def __init__(self, array, pointing, site, **kwargs):
        super().__init__(array, pointing, site)

        self.white_noise_level = kwargs.get("white_noise_level", 1e-4)

    def _run(self):
        self.data = self.white_noise_level * np.random.standard_normal(
            size=(self.array.n_dets, self.pointing.n_time)
        )


class PinkNoiseSimulation(base.BaseSimulation):
    """
    White noise! It's Gaussian.
    """

    def __init__(self, array, pointing, site, **kwargs):
        super().__init__(array, pointing, site)
        self.pink_noise_level = kwargs.get("pink_noise_level", 2.3)
        self.pink_noise_slope = kwargs.get("pink_noise_slope", 0.5)

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
        s = (f / amp) ** -(self.pink_noise_slope)
        s[np.logical_or(f < f_min, f > f_max)] = 0  # apply band pass
        return s

    def _run(self):
        dat = np.ones((self.array.n_dets, self.pointing.n_time))

        for i in range(len(dat)):
            dat[i] = self._spectrum_noise(
                spectrum_func=self._pink_spectrum,
                size=int(self.pointing.n_time),
                dt=self.pointing.time[1] - self.pointing.time[0],
            )

        self.data = dat * self.pink_noise_level
