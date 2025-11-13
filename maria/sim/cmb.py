from __future__ import annotations

import logging
import time as ttime

import dask.array as da
import healpy as hp
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..cmb import CMB, DEFAULT_CMB_KWARGS, generate_cmb, get_cmb
from ..constants import T_CMB, k_B
from ..functions.radiometry import (
    inverse_planck_spectrum,
    inverse_rayleigh_jeans_spectrum,
    planck_spectrum,
    rayleigh_jeans_spectrum,
)  # noqa
from ..io import DEFAULT_BAR_FORMAT, humanize_time
from .observation import Observation

logger = logging.getLogger("maria")


class CMBMixin:
    def _init_cmb(self, cmb: str | CMB, **cmb_kwargs):
        self.cmb_kwargs = DEFAULT_CMB_KWARGS.copy()
        self.cmb_kwargs.update(cmb_kwargs)

        if cmb in ["spectrum", "power_spectrum", "generate", "generated"]:
            for _ in tqdm(
                range(1),
                desc=f"Generating CMB (nside={self.cmb_kwargs['nside']})",
                disable=self.disable_progress_bars,
            ):
                self.cmb = generate_cmb(**self.cmb_kwargs)
        elif cmb in ["real", "planck"]:
            self.cmb = get_cmb(**self.cmb_kwargs)
        else:
            raise ValueError(f"Invalid value for cmb '{cmb}'.")

    def _compute_cmb_loading(self, obs: Observation, eps: float = 1e-6):
        cmb_loading = np.zeros(obs.shape, dtype=self.dtype)

        bands_pbar = tqdm(
            obs.instrument.bands,
            desc="Sampling CMB",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
            ncols=250,
        )

        stokes_weight = obs.instrument.dets.stokes_weight()

        for band in bands_pbar:
            bands_pbar.set_postfix(band=band.name)
            band_mask = obs.instrument.dets.band_name == band.name

            # the CMB is not a Rayleigh-Jeans source, so we do the power integrals below
            test_T_b = np.array([T_CMB, T_CMB + eps])
            test_T_RJ = inverse_rayleigh_jeans_spectrum(
                planck_spectrum(T_b=test_T_b, nu=band.nu.Hz[:, None]), nu=band.nu.Hz[:, None]
            )

            if hasattr(obs, "atmosphere"):
                opacity = sp.interpolate.interp1d(obs.atmosphere.spectrum.side_nu, obs.atmosphere.spectrum._opacity)(
                    band.nu.Hz
                )

                # the 1e12 is for picowatts
                det_power_grid = (
                    1e12
                    * k_B
                    * np.trapezoid(
                        test_T_RJ[None, None, None] * (np.exp(-opacity) * band.passband(band.nu.Hz))[..., None],
                        x=band.nu.Hz,
                        axis=-2,
                    )
                )

                P = sp.interpolate.RegularGridInterpolator(
                    obs.atmosphere.spectrum.points[:3],
                    det_power_grid,
                )((obs.atmosphere.weather.temperature[0], obs.zenith_scaled_pwv[band_mask], obs.coords.el[band_mask]))

            else:
                P = (
                    1e12
                    * k_B
                    * np.trapezoid(
                        test_T_RJ * band.passband(band.nu.Hz[:, None]),
                        x=band.nu.Hz,
                        axis=-2,
                    )
                )

            assert P.shape[-1] == 2

            pW_per_K_CMB = (P[..., 1] - P[..., 0]) / eps

            # flat_band_pixel_index = hp.ang2pix(
            #     nside=obs.cmb.nside,
            #     phi=obs.coords.l[band_mask],
            #     theta=np.pi / 2 - obs.coords.b[band_mask],
            # ).ravel()

            band_coords = obs.coords[band_mask]

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
                    pW_per_K_CMB * s * (pmat @ self.cmb.data[stokes_index, 0, 0].compute()).reshape(band_coords.shape)
                )
                logger.debug(
                    f"Computed Stokes {stokes} CMB anisotropy for band {band.name} "
                    f"in {humanize_time(ttime.monotonic() - cmb_stokes_s)}"
                )

        return da.asarray(cmb_loading, dtype=self.dtype)
