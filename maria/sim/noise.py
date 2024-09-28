import dask.array as da
import numpy as np
from tqdm import tqdm

from ..base import BaseSimulation
from ..noise import generate_noise_with_knee


class NoiseMixin:
    def _run(self):
        self._simulate_noise()

    def _simulate_noise(self):
        self.data["noise"] = da.from_array(
            np.zeros((self.instrument.n_dets, self.plan.n_time))
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


class NoiseSimulation(NoiseMixin, BaseSimulation):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(NoiseSimulation, self).__init__(*args, **kwargs)
