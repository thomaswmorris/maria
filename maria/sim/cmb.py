from __future__ import annotations

import logging
import time as ttime

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
from ..io import humanize_time

logger = logging.getLogger("maria")


class CMBMixin:
    def _simulate_cmb_emission(self, eps: float = 1e-6):
        cmb_loading = np.zeros(self.shape, dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.bands,
            desc="Sampling CMB",
            disable=self.disable_progress_bars,
        )

        stokes_weight = self.instrument.dets.stokes_weight()

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

            band_coords = self.coords[band_mask]

            pointing_s = ttime.monotonic()
            pmat = self.cmb.pointing_matrix(band_coords)
            logger.debug(
                f"Computed CMB pointing matrix for band {band.name} in {humanize_time(ttime.monotonic() - pointing_s)}"
            )

            # the total loading from the mean temperature
            loading_s = ttime.monotonic()
            cmb_loading[band_mask] = P[..., 0] * stokes_weight[band_mask, 0][:, None]
            logger.debug(f"Computed CMB loading for band {band.name} in {humanize_time(ttime.monotonic() - loading_s)}")

            for stokes_index, stokes in enumerate("IQU"):
                bands_pbar.set_postfix(band=band.name, stokes=stokes)
                cmb_stokes_s = ttime.monotonic()
                s = stokes_weight[band_mask, stokes_index][:, None]
                if np.isclose(s, 0).all():
                    logger.debug(f"Skipping stokes {stokes} for CMB (no weight)")
                    continue

                cmb_loading[band_mask] += (
                    pW_per_K_CMB * s * (self.cmb.data[stokes_index, 0].compute() @ pmat).reshape(band_coords.shape)
                )
                logger.debug(
                    f"Computed Stokes {stokes} CMB anisotropy for band {band.name} "
                    f"in {humanize_time(ttime.monotonic() - cmb_stokes_s)}"
                )

        self.loading["cmb"] = da.asarray(cmb_loading, dtype=self.dtype)
        del cmb_loading
