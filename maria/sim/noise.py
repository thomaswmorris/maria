from __future__ import annotations

import os

import dask.array as da
import numpy as np
from tqdm import tqdm

from ..io import DEFAULT_BAR_FORMAT
from ..noise import generate_noise_with_knee


class NoiseMixin:
    def _run(self):
        self._simulate_noise()

    def _simulate_noise(self, obs):
        noise_loading = np.zeros(shape=obs.shape, dtype=self.dtype)

        bands_pbar = tqdm(
            obs.instrument.dets.bands,
            desc="Generating noise",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
            ncols=250,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = obs.instrument.dets.band_name == band.name

            total_NEP = band.NEP.to("W√s") + band.NEP_per_loading.to("W√s") * sum(
                [d[band_mask].compute() for d in obs.loading.values()]
            )

            noise_loading[band_mask] = (
                1e12
                * total_NEP
                * generate_noise_with_knee(
                    shape=(band_mask.sum(), obs.plan.n),
                    sample_rate=obs.plan.sample_rate.Hz,
                    knee=band.knee,
                )
            )

        obs.loading["noise"] = da.asarray(noise_loading, dtype=self.dtype)
