from __future__ import annotations

import os

import dask.array as da
import numpy as np
from tqdm import tqdm

from ..noise import generate_noise_with_knee


class NoiseMixin:
    def _run(self):
        self._simulate_noise()

    def _simulate_noise(self):
        noise_loading = np.zeros(shape=self.shape, dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Generating noise",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            self.total_NEP = band.NEP + band.NEP_per_loading * sum([d[band_mask].compute() for d in self.loading.values()])

            noise_loading[band_mask] = (
                1e12
                * self.total_NEP
                * generate_noise_with_knee(
                    shape=(band_mask.sum(), self.plan.n_time),
                    sample_rate=self.plan.sample_rate,
                    knee=band.knee,
                )
            )

        self.loading["noise"] = da.asarray(noise_loading, dtype=self.dtype)
