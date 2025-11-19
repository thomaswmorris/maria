from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import dask.array as da
import numpy as np
import scipy as sp
from jax import jit
from jax import scipy as jsp
from tqdm import tqdm

from ..beam import compute_angular_fwhm
from ..constants import k_B
from ..io import DEFAULT_BAR_FORMAT, DEFAULT_TIME_FORMAT, fetch, humanize_time
from ..map import HEALPixMap, Map, ProjectedMap, load
from ..units import Quantity

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _initialize_map(self, map: str | Map, **map_kwargs):
        if isinstance(map, str):
            self.map = load(fetch(map), **map_kwargs)
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
                map_start = arrow.get(self.map.t.min()).to("utc")
                map_end = arrow.get(self.map.t.max()).to("utc")
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
                self.map = self.map.squeeze("t")

    def _run(self, **kwargs):
        self._sample_maps(**kwargs)

    def _sample_maps(self, obs):
        map_loading = np.zeros(obs.coords.shape, dtype=self.dtype)

        stokes_weights = obs.instrument.dets.stokes_weight()

        bands_pbar = tqdm(
            obs.instrument.dets.bands,
            desc="Sampling map",
            disable=self.disable_progress_bars,
            bar_format=DEFAULT_BAR_FORMAT,
            ncols=250,
            postfix={"band": "", "channel": "", "stokes": ""},
        )
        for band in bands_pbar:
            bands_pbar.set_postfix(band=band.name)

            band_mask = obs.instrument.dets.band_name == band.name

            band_coords = obs.coords[band_mask]

            pointing_s = ttime.monotonic()
            pmat = self.map.pointing_matrix(band_coords)
            logger.debug(f"Computed pointing matrix for band {band.name} in {humanize_time(ttime.monotonic() - pointing_s)}")

            band_fwhm = Quantity(
                compute_angular_fwhm(
                    fwhm_0=obs.instrument.dets.primary_size.mean(),
                    z=np.inf,
                    nu=band.center.Hz,
                ),
                "rad",
            )

            # ideally we would do this for each nu bin, but that's slow
            smoothed_map = self.map.smooth(fwhm=band_fwhm)

            for channel_index, (nu_min, nu_max) in enumerate(self.map.nu_bin_bounds):
                channel_map = smoothed_map.to("K_RJ", band=band).data[:, channel_index]
                qchannel = (nu_min, nu_max)
                channel_string = f"{qchannel}"

                spectrum_kwargs = (
                    {
                        "spectrum": obs.atmosphere.spectrum,
                        "zenith_pwv": obs.zenith_scaled_pwv[band_mask].compute(),
                        "base_temperature": obs.atmosphere.weather.temperature[0],
                        "elevation": obs.coords.el[band_mask],
                    }
                    if getattr(obs, "atmosphere", None)
                    else {}
                )

                calibration_s = ttime.monotonic()
                pW_per_K_RJ = (
                    1e12
                    * k_B
                    * band.compute_transmission_integral(nu_min_Hz=nu_min.Hz, nu_max_Hz=nu_max.Hz, **spectrum_kwargs)
                )
                logger.debug(
                    f"Computed K_RJ -> pW calibration for band {band.name}, channel {channel_string} in "
                    f"{humanize_time(ttime.monotonic() - calibration_s)}"
                )

                for stokes_index, stokes in enumerate(getattr(self.map, "stokes", "I")):
                    bands_pbar.set_postfix(band=band.name, channel=channel_string, stokes=stokes)

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

                    # flat_padded_map = np.pad(channel_stokes_map, pad_width=((1, 1)), mode="edge").ravel()
                    # flat_padded_map = np.pad(channel_stokes_map, pad_width=((1, 1)), mode="edge").ravel()

                    pW = pW_per_K_RJ * stokes_weight * (pmat @ channel_stokes_map.ravel()).reshape(band_coords.shape)

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

        obs.loading["map"] = da.asarray(map_loading, dtype=self.dtype)
        del map_loading
