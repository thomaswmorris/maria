from __future__ import annotations

import dask.array as da
import numpy as np
from tqdm import tqdm

from ..io import DEFAULT_BAR_FORMAT
from ..noise import generate_noise_with_knee
from ..utils import compute_diameter, generate_spatial_basis

DEFAULT_NOISE_SIM_KWARGS = {"correlated_noise_proportion": 0.5, "correlated_noise_spatial_scale": 1.0}


class NoiseMixin:
    def _run(self):
        self._simulate_noise()

    def _simulate_noise(self, obs):

        noise_loading = np.zeros(shape=obs.shape, dtype=self.dtype)
        dets = obs.instrument.dets

        bands_pbar = tqdm(
            dets.bands,
            desc="Generating noise",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
            ncols=250,
        )

        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = dets.band_name == band.name

            total_NEP = dets.NEP[band_mask][:, None] + band.NEP_per_loading.to("W√s") * sum(
                [d[band_mask].compute() for d in obs.loading.values()]
            )

            band_offsets = dets.offsets[band_mask]
            fov = compute_diameter(band_offsets)

            if (fov > 0) and (band_mask.sum() > 16):
                basis = generate_spatial_basis(
                    offsets=band_offsets,
                    k=5,
                    n_side=16,
                    scale=fov * self.noise_kwargs.get("correlated_noise_spatial_scale", 0),
                )
            else:
                basis = np.ones((band_mask.sum(), 1))

            unscaled_noise = generate_noise_with_knee(
                shape=(band_mask.sum(), obs.plan.n),
                sample_rate=obs.plan.sample_rate.Hz,
                knee=band.knee,
                basis=basis,
                corr_prop=self.noise_kwargs.get("correlated_noise_proportion", 0),
            )

            # put her in picowatts
            noise_loading[band_mask] = 1e12 * total_NEP * unscaled_noise

        obs.loading["noise"] = da.asarray(noise_loading, dtype=self.dtype)
