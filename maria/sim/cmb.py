from __future__ import annotations

import dask.array as da
import healpy as hp
import numpy as np
from tqdm import tqdm


class CMBMixin:
    def _simulate_cmb_emission(self):

        self.loading["cmb"] = da.zeros(
            shape=(self.instrument.n_dets, self.plan.n_time), dtype=self.dtype
        )

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

            band_cmb_temperature_samples = self.cmb.data[0, 0, 0][
                flat_band_pixel_index
            ].reshape(band_mask.sum(), -1)

            if hasattr(self, "atmosphere"):
                band_cal = band.cal(
                    signature=f"{self.cmb.units} -> pW",
                    spectrum=self.atmosphere.spectrum,
                    zenith_pwv=self.zenith_scaled_pwv[band_mask],
                    base_temperature=self.atmosphere.weather.temperature[0],
                    elevation=np.degrees(self.coords.el[band_mask]),
                )
            else:
                band_cal = band.cal(f"{self.cmb.units} -> pW")

            self.loading["cmb"][band_mask] = band_cal(
                band_cmb_temperature_samples.compute()
            )

            # P_lo, P_hi = (
            #     1e12
            #     * k_B
            #     * np.trapezoid(
            #         y=cmb_rj_spectrum * band.passband(test_nu), x=1e9 * test_nu
            #     )
            # )  # noqa
            # band_cmb_power_map = (cmb_anisotropies_uK - T_lo) * (P_hi - P_lo) / (
            #     T_hi - T_lo
            # ) + P_lo
            # self.data["cmb"][band_mask] = band_cmb_power_map[:, flat_band_pixel_index][
            #     0
            # ].reshape(band_mask.sum(), -1)
