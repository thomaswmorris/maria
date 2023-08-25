import numpy as np

from . import base, utils

class NoiseSimulation(base.BaseSimulation):
    """
    The base class for modeling noise.
    """
    def __init__(self, array, pointing, site, **kwargs):
        super().__init__(array, pointing, site)

        self.white_noise_level = kwargs.get("white_noise", 1)

    def _run(self):

        self.temperature = self.white_noise_level * np.random.standard_normal(size=(self.pointing.nt, self.pointing.nt))
              