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


class OneOverEffNoiseSimulation(base.BaseSimulation):
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
