from __future__ import annotations

import logging
import os
import time as ttime

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from maria.io import humanize_time

from ..constants import k_B

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")


class AtmosphereMixin:
    def _simulate_atmosphere(self):
        # this produces self.atmosphere.zenith_scaled_pwv at lower res
        # which we use to compute emission and opacity
        self.atmosphere.simulate_pwv()

        # update to the sim resolution
        self.zenith_scaled_pwv = da.from_array(
            sp.interpolate.interp1d(
                self.atmosphere.coords.t,
                self.atmosphere.zenith_scaled_pwv,
                bounds_error=False,
                fill_value="extrapolate",
            )(self.coords.t),
        )

    def _compute_atmospheric_emission(self):
        self.loading["atmosphere"] = da.zeros_like(self.zenith_scaled_pwv)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Computing atmospheric emission",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            start_s = ttime.monotonic()

            bands_pbar.set_postfix({"band": band.name})

            band_index = self.instrument.dets.mask(band_name=band.name)

            # in picowatts. the 1e9 is for GHz -> Hz
            det_power_grid = (
                1e12
                * k_B
                * np.trapezoid(
                    self.atmosphere.spectrum._emission * band.passband(self.atmosphere.spectrum.side_nu),
                    self.atmosphere.spectrum.side_nu,
                    axis=-1,
                )
            )

            band_power_interpolator = sp.interpolate.RegularGridInterpolator(
                self.atmosphere.spectrum.points[:3],
                det_power_grid,
            )

            self.loading["atmosphere"][band_index] = band_power_interpolator(
                (
                    self.atmosphere.weather.temperature[0],
                    self.zenith_scaled_pwv[band_index],
                    self.coords.el[band_index],
                ),
            )

            logger.debug(f"Sampled atmosphere for band {band.name} in {humanize_time(ttime.monotonic() - start_s)}.")
