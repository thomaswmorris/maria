from __future__ import annotations

import dask.array as da
from tqdm import tqdm

from ..noise import generate_noise_with_knee


class NoiseMixin:
    def _run(self):
        self._simulate_noise()

    def _simulate_noise(self):
        self.loading["noise"] = da.zeros(shape=(self.instrument.n_dets, self.plan.n_time), dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Generating noise",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            self.total_NEP = band.NEP + band.NEP_per_loading * self.total_loading[band_mask]

            self.loading["noise"][band_mask] = (
                1e12
                * self.total_NEP
                * generate_noise_with_knee(
                    self.plan.time,
                    n=band_mask.sum(),
                    knee=band.knee,
                )
            )
