from __future__ import annotations

import dask.array as da
import healpy as hp
import numpy as np
from tqdm import tqdm

from ..functions import planck_spectrum, inverse_rayleigh_jeans_spectrum
from ..constants import T_CMB, k_B


class CMBMixin:
    def _simulate_cmb_emission(self):

        cmb_anisotropies_uK = T_CMB + self.cmb.data[:, 0, 0]

        T_lo = cmb_anisotropies_uK.min().compute()
        T_hi = cmb_anisotropies_uK.max().compute()

        test_cmb_anisotropies_uK = np.array([T_lo, T_hi])  # noqa

        test_nu = np.linspace(1e0, 5e2, 256)
        cmb_brightness = planck_spectrum(
            T_CMB + 1e-6 * test_cmb_anisotropies_uK[:, None],
            1e9 * test_nu,
        )

        cmb_rj_spectrum = inverse_rayleigh_jeans_spectrum(
            cmb_brightness, nu=1e9 * test_nu
        )

        self.data["cmb"] = da.zeros(
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
            ).ravel()  # noqa
            P_lo, P_hi = (
                1e12
                * k_B
                * np.trapezoid(
                    y=cmb_rj_spectrum * band.passband(test_nu), x=1e9 * test_nu
                )
            )  # noqa
            band_cmb_power_map = (cmb_anisotropies_uK - T_lo) * (P_hi - P_lo) / (
                T_hi - T_lo
            ) + P_lo
            self.data["cmb"][band_mask] = band_cmb_power_map[:, flat_band_pixel_index][
                0
            ].reshape(band_mask.sum(), -1)
