from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import dask.array as da
import numpy as np
import scipy as sp
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import jit
from jax import scipy as jsp
from jax.experimental import sparse as jsparse
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

from ..beam import compute_angular_fwhm
from ..constants import k_B
from ..io import DEFAULT_BAR_FORMAT, DEFAULT_TIME_FORMAT, humanize_time
from ..map import Map, get
from ..units import Quantity

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

@jax.jit(static_argnames=["band_coords_shape"])
def apply_pointing(P, map_vec, pW_per_K_RJ, band_coords_shape):
    return pW_per_K_RJ * (P @ map_vec).reshape(band_coords_shape)

@jax.jit(static_argnames=["band_coords_shape"])
def _sample_maps_jax(
    map_loading,
    band_coords_shape,
    pW_per_K_RJ_all,
    P_all,
    map_vec_all,
    band_indices
):

    def band_step(map_loading, inputs):
        band_idx = inputs

        for i in range(len(pW_per_K_RJ_all)):
            pW_per_K_RJ = pW_per_K_RJ_all[i]
            P = P_all[i]
            map_vec = map_vec_all[i]

            pW = apply_pointing(P, map_vec, pW_per_K_RJ, band_coords_shape)
            map_loading = map_loading.at[band_idx].add(pW)

        return map_loading, None

    map_loading, _ = lax.scan(band_step, map_loading, band_indices)

    return map_loading

DEFAULT_MAP_SIM_KWARGS = {
    "bilinear_sampling": True,
}


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _initialize_map(self, map: str | Map, **map_kwargs):
        if isinstance(map, str):
            self.map = get(map, **map_kwargs)
        elif isinstance(map, Map):
            self.map = map
        else:
            raise ValueError("'map' must be either a Map or a string")

        # the map can be frequency-naive if it is already in K_RJ
        # self.map = map.to(units="K_RJ")

        if "stokes" not in self.map.dims:
            self.map = self.map.unsqueeze("stokes")

        if "nu" not in self.map.dims:
            self.map = self.map.unsqueeze("nu")

        if "t" in self.map.dims:
            if self.map.dims["t"] > 1:
                map_start = arrow.get(self.map.t.seconds.min()).to("utc")
                map_end = arrow.get(self.map.t.seconds.max()).to("utc")
                if map_start > self.min_time:
                    logger.warning(
                        f"Beginning of map ({map_start.format(DEFAULT_TIME_FORMAT)}) is after the "
                        f"beginning of the simulation ({self.min_time.format(DEFAULT_TIME_FORMAT)}).",
                    )
                if map_end < self.max_time:
                    logger.warning(
                        f"End of map ({map_end.format(DEFAULT_TIME_FORMAT)}) is before the "
                        f"end of the simulation ({self.max_time.format(DEFAULT_TIME_FORMAT)}).",
                    )
        else:
            self.map = self.map.unsqueeze("t")

    def _run(self, **kwargs):
        self._sample_maps(**kwargs)

    def _sample_maps(self, obs):
        map_loading = jnp.zeros(obs.coords.shape, dtype=self.dtype)
        instrument = obs.instrument
        disable_progress_bars = self.disable_progress_bars
        coords = obs.coords
        map = self.map
        atmosphere = obs.atmosphere

        bands_pbar = tqdm(
            instrument.dets.bands,
            desc="Sampling map",
            disable=disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
            ncols=250,
            postfix={"band": "", "channel": "", "stokes": ""},
        )

        # store values for jax computation
        pW_per_K_RJ_all = []
        P_all = []
        map_vec_all = []
        band_index_list = []

        for band in bands_pbar:
            band_mask = instrument.dets.band_name == band.name
            indices = np.where(band_mask)[0]
            band_index_list.append(indices)

            band_dets = instrument.dets[band_mask]
            band_coords = coords[band_mask]

            band_fwhm = Quantity(
                compute_angular_fwhm(
                    fwhm_0=instrument.dets.primary_size.mean(),
                    z=np.inf,
                    nu=band.center.Hz,
                ),
                "rad",
            )

            # input for forward model is smoothed map and not unsmoothed
            smoothed_map = map.smooth(fwhm=band_fwhm)
            logger.debug(f"Convolved map with beam width {band_fwhm} for band {band.name}")

            for channel_index, (nu_min, nu_max) in enumerate(map.nu_bin_bounds):
                channel_map = smoothed_map.to("K_RJ", band=band)[:, [channel_index]]
                qchannel = (nu_min, nu_max)
                channel_string = f"{qchannel}"

                if (band.nu.Hz.max() < nu_min.Hz) or (nu_max.Hz < band.nu.Hz.min()):
                    continue

                spectrum_kwargs = (
                    {
                        "spectrum": atmosphere.spectrum,
                        "zenith_pwv": obs.zenith_scaled_pwv[band_mask].compute(),
                        "base_temperature": atmosphere.weather.temperature[0],
                        "elevation": coords.el[band_mask],
                    }
                    if getattr(obs, "atmosphere", None)
                    else {}
                )
                calibration_s = ttime.monotonic()
                pW_per_K_RJ = jnp.array(
                    1e12
                    * k_B
                    * band.compute_transmission_integral(nu_min_Hz=nu_min.Hz, nu_max_Hz=nu_max.Hz, **spectrum_kwargs)
                )
                pW_per_K_RJ_all.append(pW_per_K_RJ)
                logger.debug(
                    f"Computed K_RJ -> pW calibration for band {band.name}, channel {channel_string} in "
                    f"{humanize_time(ttime.monotonic() - calibration_s)}"
                )
                bands_pbar.set_postfix(band=band.name, channel=channel_string)

                pointing_s = ttime.monotonic()

                P = channel_map.stokes_weighted_pointing_matrix(
                    coords=band_coords,
                    dets=band_dets,
                    bilinear=self.map_kwargs.get("bilinear_sampling", DEFAULT_MAP_SIM_KWARGS["bilinear_sampling"]),
                )
                P = jsparse.BCSR.from_scipy_sparse(P)
                logger.debug(
                    f"Computed pointing matrix for band {band.name} in {humanize_time(ttime.monotonic() - pointing_s)}"
                )
                P_all.append(P)
                map_vec_all.append(jnp.asarray(channel_map.data.ravel()))

        band_indices = jnp.asarray(band_index_list)
        map_loading = _sample_maps_jax(map_loading, band_coords.shape, pW_per_K_RJ_all, P_all, map_vec_all, band_indices)
        obs.loading["map"] = da.asarray(map_loading, dtype=self.dtype)
        del map_loading
