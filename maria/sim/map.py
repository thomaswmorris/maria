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
from ..calibration import Calibration
from ..constants import k_B
from ..io import DEFAULT_BAR_FORMAT, DEFAULT_TIME_FORMAT, humanize_time
from ..map import HEALPixMap, Map, get
from ..units import Quantity

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

DEFAULT_MAP_SIM_KWARGS = {
    "bilinear_sampling": False,
}


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _initialize_map(self, map: str | Map, **map_kwargs):
        if isinstance(map, str):
            m = get(map, **map_kwargs)
        elif isinstance(map, Map):
            m = map.copy()
        else:
            raise ValueError("'map' must be either a Map or a string")

        # the map can be frequency-naive if it is already in K_RJ
        # self.map = map.to(units="K_RJ")

        if "stokes" not in m.dims:
            m = m.unsqueeze("stokes")

        if "nu" not in m.dims:
            m = m.unsqueeze("nu")

        if "t" in m.dims:
            if m.dims["t"] > 1:
                map_start = arrow.get(m.t.seconds.min()).to("utc")
                map_end = arrow.get(m.t.seconds.max()).to("utc")
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
            m = m.unsqueeze("t")

        parity_signature = {dim: (-1 if dim in ["v", "eta"] else 1) for dim in m.dims}
        m.apply_parity(**parity_signature)

        self.maps["map"] = m

    def _run(self, **kwargs):
        self._sample_maps(**kwargs)

    def _sample_maps(self, obs):

        for map_name, m in self.maps.items():
            map_loading = np.zeros(obs.coords.shape, dtype=self.dtype)

            bands_pbar = tqdm(
                obs.instrument.dets.bands,
                desc=f"Sampling source '{map_name}'",
                disable=self.disable_progress_bars,
                bar_format=DEFAULT_BAR_FORMAT,
                ncols=250,
                postfix={"band": "", "channel": "", "stokes": ""},
            )
            for band in bands_pbar:
                bands_pbar.set_postfix(band=band.name)

                band_mask = obs.instrument.dets.band_name == band.name
                band_coords = obs.coords[band_mask]
                band_dets = obs.instrument.dets[band_mask]

                band_fwhm = Quantity(
                    compute_angular_fwhm(
                        fwhm_0=obs.instrument.dets.primary_size.mean(),
                        z=np.inf,
                        nu=band.center.Hz,
                    ),
                    "rad",
                )

                # ideally we would do this for each nu bin, but that's slow

                if not isinstance(m, HEALPixMap):
                    smoothed_map = m.smooth(fwhm=band_fwhm)
                else:
                    smoothed_map = m

                logger.debug(f"Convolved map with beam width {band_fwhm} for band {band.name}")

                bands_pbar.set_postfix(band=band.name, message=f"Computing pointing matrix")

                pointing_s = ttime.monotonic()

                P = m[:, 0].stokes_weighted_pointing_matrix(
                    coords=band_coords,
                    dets=band_dets,
                    bilinear=self.map_kwargs.get("bilinear_sampling", DEFAULT_MAP_SIM_KWARGS["bilinear_sampling"]),
                )
                logger.debug(
                    f"Computed pointing matrix for band {band.name} in {humanize_time(ttime.monotonic() - pointing_s)}"
                )

                for channel_index, (nu_min, nu_max) in enumerate(m.nu_bin_bounds):
                    if (band.nu.Hz.max() < nu_min) or (nu_max < band.nu.Hz.min()):
                        continue

                    nu_min = max(band.nu.Hz.min(), nu_min)
                    nu_max = min(band.nu.Hz.max(), nu_max)

                    channel_s = ttime.monotonic()
                    channel_map = smoothed_map[:, channel_index]
                    channel_map_data = channel_map.data.compute()
                    channel_string = f"{(Quantity(nu_min, 'Hz'), Quantity(nu_max, 'Hz'))}"

                    bands_pbar.set_postfix(band=band.name, message=f"Sampling channel {channel_string}")

                    calibration_s = ttime.monotonic()

                    if hasattr(obs, "atmosphere"):
                        nu = obs.atmosphere.spectrum.side_nu
                        channel_nu_mask = (nu >= nu_min) & (nu <= nu_max)
                        channel_nu = nu[channel_nu_mask]

                        instrument_transmission = band.passband(channel_nu)

                        static_atmospheric_transmission = obs.atmosphere.spectrum.transmission(
                            nu=channel_nu,
                            pwv=obs.zenith_scaled_pwv[band_mask].mean().compute(),
                            elevation=obs.coords.el[band_mask].mean(),
                        )

                        static_transmission = instrument_transmission * static_atmospheric_transmission

                        f_grid = (
                            instrument_transmission
                            * np.exp(-obs.atmosphere.spectrum._opacity[..., channel_nu_mask])
                            * static_transmission
                        ).sum(axis=-1) / (static_transmission**2).sum(axis=-1)

                        xi = (
                            obs.atmosphere.weather.temperature[0],
                            obs.zenith_scaled_pwv[band_mask].compute(),
                            obs.coords.el[band_mask],
                        )

                        f = jsp.interpolate.RegularGridInterpolator(
                            points=obs.atmosphere.spectrum.points[:3], values=f_grid
                        )(xi)

                    else:
                        channel_nu = np.linspace(band.nu.Hz.min(), band.nu.Hz.max(), 100)
                        static_atmospheric_transmission = np.zeros_like(channel_nu)

                        static_transmission = band.passband(channel_nu)
                        f = 1

                    # computing the spectrum for every map pixel would be silly
                    # compute the power for the smallest and largest values, and then interpolate!
                    map_values = np.nanpercentile(
                        channel_map_data, weights=channel_map.weight.compute(), method="inverted_cdf", q=[0, 100]
                    )

                    # the calibration might need these to compute T_RJ, so give them just to be safe
                    map_calibration_kwargs = {"pixel_area": channel_map.pixel_area.sr, "beam_area": channel_map.beam_area.sr}

                    map_values_trj_spectrum = Calibration(f"{channel_map.units} -> K_RJ", **map_calibration_kwargs)(
                        map_values[:, None], nu=channel_nu
                    )

                    logger.debug(
                        f"Computed {channel_map.units} -> pW calibration for band {band.name}, channel {channel_string} in "
                        f"{humanize_time(ttime.monotonic() - calibration_s)}"
                    )

                    map_values_power = k_B * np.trapezoid(map_values_trj_spectrum * static_transmission, x=channel_nu)

                    channel_power_map = np.interp(channel_map_data, map_values, map_values_power)

                    pW = 1e12 * f * (P @ channel_power_map.ravel()).reshape(band_coords.shape)

                    map_loading[band_mask] += pW

                    if logger.level == 10:
                        logger.debug(
                            f"Computed map load {pW.shape} ~{Quantity(pW.std(), 'pW'):<8} for band {band.name}, "
                            f"channel {channel_string} in {humanize_time(ttime.monotonic() - channel_s)}"
                        )

                    del pW

                if map_loading[band_mask].sum() == 0:
                    logger.warning(f"No load from map for band {band.name}")

            # the above samples the map instantaneously
            # so you get all the lower noise per sample benefits without the smearing
            # obviously we can't continuum sample, so we convolve the map loading with a triangular kernel
            map_loading = sp.ndimage.convolve1d(map_loading, weights=np.array([0.25, 0.5, 0.25]), axis=-1)

            obs.loading[map_name] = da.asarray(map_loading, dtype=self.dtype)
