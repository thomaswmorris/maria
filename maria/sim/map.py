from __future__ import annotations

import logging
import os
import time as ttime

import dask.array as da
import numpy as np
import scipy as sp
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
        dx, dy = self.coords.offsets(frame=self.map.frame, center=self.map.center, compute=True)

        self.loading["map"] = da.zeros_like(dx, dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Sampling map",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            band_start_s = ttime.monotonic()

            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            band_fwhm = compute_angular_fwhm(
                fwhm_0=self.instrument.dets.primary_size.mean(),
                z=np.inf,
                nu=band.center,
            )

            # ideally we would do this for each nu bin, but that's slow
            smoothed_map = self.map.smooth(fwhm=band_fwhm)

            for nu_index, (nu_min, nu_max) in enumerate(self.map.nu_bin_bounds):
                # TODO: skip if the band can't see the map nu bin

                qnu_min = Quantity(nu_min, 'Hz')
                qnu_max = Quantity(nu_max, 'Hz')

                spectrum_kwargs = (
                    {
                        "spectrum": self.atmosphere.spectrum,
                        "zenith_pwv": self.zenith_scaled_pwv[band_mask],
                        "base_temperature": self.atmosphere.weather.temperature[0],
                        "elevation": self.coords.el[band_mask],
                    }
                    if hasattr(self, "atmosphere")
                    else {}
                )

                sample_integral = band.compute_nu_integral(nu_min=nu_min, nu_max=nu_max, **spectrum_kwargs)
                # channel_gain = 1e12 * k_B * sample_integral

                # logger.debug(f"Map bandwidth ({qnu_min} - {qnu_max}) has gain {channel_gain:.02e} pW/K_RJ")

                if len(self.map.t) > 1:
                    sample_T_RJ = sp.interpolate.RegularGridInterpolator(
                        (self.map.t, self.map.y_side, self.map.x_side),
                        smoothed_map.data[0, nu_index].compute(),
                        bounds_error=False,
                        fill_value=0,
                        method="linear",
                    )((self.boresight.t, dy[band_mask], dx[band_mask]))

                else:
                    # # dx, dy = self.coords.offsets(frame=self.map.frame, center=self.map.center)

                    # a = ttime.monotonic()

                    # ix = np.digitize(dx[band_mask], bins=self.map.x_bins)
                    # iy = np.digitize(dy[band_mask], bins=self.map.y_bins)
                    # padded_map = np.pad(smoothed_map.data[0, nu_index, 0].compute(), [(1, 1), (1, 1)], mode='edge')
                    # sample_T_RJ = padded_map[iy, ix]

                    # b = ttime.monotonic()

                    m = smoothed_map.data[0, nu_index, 0].compute()
                    edge_value = np.median(np.c_[m[0], m[-1], m[:, 0], m[:, -1]])

                    sample_T_RJ = sp.interpolate.RegularGridInterpolator(
                        (self.map.y_side, self.map.x_side),
                        m,
                        bounds_error=False,
                        fill_value=edge_value,
                        method="linear",
                    )((dy[band_mask], dx[band_mask]))

                    # c = ttime.monotonic()

                    # print(b - a)
                    # print(c - b)

                rms_T_RJ = sample_T_RJ.std()

                if rms_T_RJ == 0:
                    logger.warning("No power from map!")

                logger.debug(f"Sampled temperature map has rms {rms_T_RJ}")

                # 1e12 because it's in picowatts
                self.loading["map"][band_mask] += 1e12 * k_B * sample_integral * sample_T_RJ

            logger.debug(f"Computed map power for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

            if self.loading["map"][band_mask].sum().compute() == 0:
                logger.warning(f"No power from map for band {band}.")

                # things that are blackbodies that we can see are

                # filtered_band_power_map = (
                #     self.map.smooth(fwhm=band_fwhm).power(band).compute()
                # )
