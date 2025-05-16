from __future__ import annotations

import logging
import os
import time as ttime

import dask.array as da
import numpy as np
import scipy as sp
from jax import jit
from jax import scipy as jsp
from tqdm import tqdm

from ..beam import compute_angular_fwhm
from ..constants import k_B
from ..io import humanize_time
from ..units import Quantity

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _run(self, **kwargs):
        self._sample_maps(**kwargs)

    def _sample_maps(self):
        map_loading = np.zeros(self.coords.shape, dtype=self.dtype)

        stokes_weights = self.instrument.dets.stokes_weight()

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Sampling map",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix(band=band.name)

            band_mask = self.instrument.dets.band_name == band.name

            band_coords = self.coords[band_mask]

            pointing_s = ttime.monotonic()
            pmat = self.map.pointing_matrix(band_coords)
            logger.debug(f"Computed pointing matrix for band {band.name} in {humanize_time(ttime.monotonic() - pointing_s)}")

            band_fwhm = compute_angular_fwhm(
                fwhm_0=self.instrument.dets.primary_size.mean(),
                z=np.inf,
                nu=band.center,
            )

            # ideally we would do this for each nu bin, but that's slow
            smoothed_map = self.map.smooth(fwhm=band_fwhm)

            for channel_index, (nu_min, nu_max) in enumerate(self.map.nu_bin_bounds):
                channel_map = smoothed_map.data[:, channel_index]
                qchannel = Quantity([nu_min, nu_max], "Hz")
                channel_string = f"{qchannel}"

                bands_pbar.set_postfix(channel=channel_string)

                spectrum_kwargs = (
                    {
                        "spectrum": self.atmosphere.spectrum,
                        # "zenith_pwv": self.zenith_scaled_pwv[band_mask].compute(),
                        "zenith_pwv": self.zenith_scaled_pwv[band_mask].mean().compute(),
                        "base_temperature": self.atmosphere.weather.temperature[0],
                        "elevation": self.coords.el[band_mask],
                    }
                    if hasattr(self, "atmosphere")
                    else {}
                )

                calibration_s = ttime.monotonic()
                pW_per_K_RJ = 1e12 * k_B * band.compute_nu_integral(nu_min=nu_min, nu_max=nu_max, **spectrum_kwargs)
                logger.debug(
                    f"Computed K_RJ -> pW calibration for band {band.name}, channel {channel_string} in "
                    f"{humanize_time(ttime.monotonic() - calibration_s)}"
                )

                for stokes_index, stokes in enumerate(getattr(self.map, "stokes", "I")):
                    stokes_weight = stokes_weights[band_mask, "IQUV".index(stokes), None]
                    if np.isclose(stokes_weight, 0).all():
                        logger.debug(f"Skipping stokes {stokes} (no weight)")
                        continue

                    stokes_channel_s = ttime.monotonic()
                    channel_stokes_map = channel_map[stokes_index].compute()

                    # sample_T_RJ = jit(jsp.interpolate.RegularGridInterpolator(
                    #             (self.map.y_side[::-1], self.map.x_side),
                    #             channel_stokes_map[::-1],
                    #             bounds_error=False,
                    #             fill_value=0,
                    #             method="linear",
                    #         ))((dy[band_mask], dx[band_mask]))

                    flat_padded_map = np.pad(channel_stokes_map, pad_width=((1, 1)), mode="edge").ravel()

                    pW = pW_per_K_RJ * stokes_weight * (flat_padded_map @ pmat).reshape(band_coords.shape)

                    map_loading[band_mask] += pW

                    if logger.level == 10:
                        logger.debug(
                            f"Computed map load {pW.shape} ~{Quantity(pW.std(), 'pW'):<8} for band {band.name}, "
                            f"channel {channel_string}, stokes {stokes} in "
                            f"{humanize_time(ttime.monotonic() - stokes_channel_s)}"
                        )

                    del pW

            if map_loading[band_mask].sum() == 0:
                logger.warning(f"No load from map for band {band.name}")

        self.loading["map"] = da.asarray(map_loading, dtype=self.dtype)
        del map_loading
