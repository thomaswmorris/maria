from __future__ import annotations

import logging
import os
import time as ttime

import dask.array as da
import numpy as np
import scipy as sp
from jax import scipy as jsp
from tqdm import tqdm

from ..constants import k_B
from ..io import DEFAULT_BAR_FORMAT, humanize_time

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")


class AtmosphereMixin:
    def _simulate_atmosphere(self, obs):
        # this produces self.atmosphere.zenith_scaled_pwv at lower res
        # which we use to compute emission and opacity
        obs.atmosphere.simulate_pwv(instrument=obs.instrument)

        # upsample to the sim resolution
        obs.zenith_scaled_pwv = da.from_array(
            sp.interpolate.interp1d(
                obs.atmosphere.coords.t,
                obs.atmosphere.zenith_scaled_pwv,
                bounds_error=False,
                fill_value="extrapolate",
            )(obs.coords.t),
        )

    def _compute_atmospheric_loading(self, obs):
        atmosphere_loading = np.zeros(obs.atmosphere.zenith_scaled_pwv.shape, dtype=self.dtype)

        bands_pbar = tqdm(
            obs.instrument.dets.bands,
            desc="Computing atmospheric emission",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
            ncols=250,
        )
        for band in bands_pbar:
            emission_s = ttime.monotonic()

            bands_pbar.set_postfix({"band": band.name})

            band_index = obs.instrument.dets.mask(band_name=band.name)

            # the 1e12 is for picowatts
            det_power_grid = (
                1e12
                * k_B
                * np.trapezoid(
                    obs.atmosphere.spectrum._emission * band.passband(obs.atmosphere.spectrum.side_nu),
                    obs.atmosphere.spectrum.side_nu,
                    axis=-1,
                )
            )

            band_power_interpolator = jsp.interpolate.RegularGridInterpolator(
                obs.atmosphere.spectrum.points[:3],
                det_power_grid,
            )

            atmosphere_loading[band_index] = band_power_interpolator(
                (
                    obs.atmosphere.weather.temperature[0],
                    obs.atmosphere.zenith_scaled_pwv[band_index],
                    obs.atmosphere.coords.el[band_index].clip(max=np.pi / 2),
                ),
            )

            logger.debug(
                f"Computed atmospheric emission for band {band.name} in {humanize_time(ttime.monotonic() - emission_s)}."
            )

        # upsample to the sim resolution
        loading = da.asarray(
            sp.interpolate.interp1d(
                obs.atmosphere.coords.t,
                atmosphere_loading,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(obs.coords.t),
            dtype=self.dtype,
        )

        return loading
