from __future__ import annotations

import dask.array as da
import healpy as hp
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..constants import T_CMB, k_B
from ..functions.radiometry import (
    inverse_planck_spectrum,
    inverse_rayleigh_jeans_spectrum,
    planck_spectrum,
    rayleigh_jeans_spectrum,
)  # noqa


class CMBMixin:
    def _simulate_cmb_emission(self, eps: float = 1e-6):
        self.loading["cmb"] = da.zeros(shape=(self.instrument.n_dets, self.plan.n_time), dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.bands,
            desc="Sampling CMB",
            disable=self.disable_progress_bars,
        )

        m = self.instrument.dets.stokes_weight()

        for band in bands_pbar:
            bands_pbar.set_postfix(band=band.name)
            band_mask = self.instrument.dets.band_name == band.name

            # the CMB is not a Rayleigh-Jeans source, so we do the power integrals below
            test_T_b = np.array([T_CMB, T_CMB + eps])
            test_T_RJ = inverse_rayleigh_jeans_spectrum(
                planck_spectrum(T_b=test_T_b, nu=band.nu[:, None]), nu=band.nu[:, None]
            )

            if hasattr(self, "atmosphere"):
                opacity = sp.interpolate.interp1d(self.atmosphere.spectrum.side_nu, self.atmosphere.spectrum._opacity)(
                    band.nu
                )

                # the 1e12 is for picowatts
                det_power_grid = (
                    1e12
                    * k_B
                    * np.trapezoid(
                        test_T_RJ[None, None, None] * (np.exp(-opacity) * band.passband(band.nu))[..., None],
                        x=band.nu,
                        axis=-2,
                    )
                )

                P = sp.interpolate.RegularGridInterpolator(
                    self.atmosphere.spectrum.points[:3],
                    det_power_grid,
                )((self.atmosphere.weather.temperature[0], self.zenith_scaled_pwv[band_mask], self.coords.el[band_mask]))

            else:
                P = (
                    1e12
                    * k_B
                    * np.trapezoid(
                        test_T_RJ * band.passband(band.nu[:, None]),
                        x=band.nu,
                        axis=-2,
                    )
                )

            assert P.shape[-1] == 2

            pW_per_K_CMB = (P[..., 1] - P[..., 0]) / eps

            flat_band_pixel_index = hp.ang2pix(
                nside=self.cmb.nside,
                phi=self.coords.l[band_mask],
                theta=np.pi / 2 - self.coords.b[band_mask],
            ).ravel()

            # the total loading from the mean temperature
            self.loading["cmb"][band_mask] += m[band_mask, 0][:, None] * P[..., 0]

            band_cmb_temperature_samples = (
                self.cmb.data[:, 0, flat_band_pixel_index].reshape(3, band_mask.sum(), -1).compute()
            )

            for stokes_index, stokes in enumerate("IQU"):
                bands_pbar.set_postfix(band=band.name, stokes=stokes)
                stokes_weight = m[band_mask, stokes_index][:, None]
                self.loading["cmb"][band_mask] += stokes_weight * pW_per_K_CMB * band_cmb_temperature_samples[stokes_index]
