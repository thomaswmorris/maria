from __future__ import annotations

import dask.array as da
import healpy as hp
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..functions import planck_spectrum, inverse_rayleigh_jeans_spectrum
from ..constants import T_CMB, k_B


class CMBMixin:
    def _simulate_cmb_emission(self):

        pixel_index = hp.ang2pix(
            nside=self.cmb.nside,
            phi=self.coords.l,
            theta=np.pi / 2 - self.coords.b,
        ).compute()  # noqa
        cmb_temperatures = self.cmb.T[pixel_index]

        test_nu = np.linspace(1e0, 5e2, 256)

        cmb_temperature_samples_K = T_CMB + np.linspace(
            self.cmb.T.min(),
            self.cmb.T.max(),
            3,
        )  # noqa
        cmb_brightness = planck_spectrum(
            cmb_temperature_samples_K[:, None],
            1e9 * test_nu,
        )

        self.data["cmb"] = da.zeros_like(
            np.empty((self.instrument.dets.n, self.plan.n_time)),
        )

        bands_pbar = tqdm(
            self.instrument.bands,
            desc="Sampling CMB",
            disable=self.disable_progress_bars,
        )

        for band in bands_pbar:

            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            T_RJ = inverse_rayleigh_jeans_spectrum(cmb_brightness, nu=1e9 * test_nu)
            band_cmb_power_samples_pW = (
                1e12
                * k_B
                * np.trapezoid(y=T_RJ * band.passband(test_nu), x=1e9 * test_nu)
            )

            self.data["cmb"][band_mask] = sp.interpolate.interp1d(
                cmb_temperature_samples_K,
                band_cmb_power_samples_pW,
            )(T_CMB + cmb_temperatures[band_mask])
