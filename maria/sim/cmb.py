from __future__ import annotations

import dask.array as da
import healpy as hp
import numpy as np
from tqdm import tqdm


class CMBMixin:
    def _simulate_cmb_emission(self):
        self.loading["cmb"] = da.zeros(shape=(self.instrument.n_dets, self.plan.n_time), dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.bands,
            desc="Sampling CMB",
            disable=self.disable_progress_bars,
        )

        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            flat_band_pixel_index = hp.ang2pix(
                nside=self.cmb.nside,
                phi=self.coords.l[band_mask],
                theta=np.pi / 2 - self.coords.b[band_mask],
            ).ravel()

            band_cmb_temperature_samples = self.cmb.data[0, 0, 0][flat_band_pixel_index].reshape(band_mask.sum(), -1)

            if hasattr(self, "atmosphere"):
                band_cal = band.cal(
                    signature=f"{self.cmb.units} -> pW",
                    spectrum=self.atmosphere.spectrum,
                    zenith_pwv=self.zenith_scaled_pwv[band_mask],
                    base_temperature=self.atmosphere.weather.temperature[0],
                    elevation=self.coords.el[band_mask],
                )
            else:
                band_cal = band.cal(f"{self.cmb.units} -> pW")

            self.loading["cmb"][band_mask] = band_cal(band_cmb_temperature_samples.compute())
