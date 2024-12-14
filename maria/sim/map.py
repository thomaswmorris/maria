from __future__ import annotations

import logging
import os

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..instrument import beam
from ..constants import k_B

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
        dx, dy = self.coords.offsets(frame=self.map.frame, center=self.map.center)

        self.data["map"] = da.from_array(
            1e-16 * np.random.standard_normal(size=dx.shape),
        )

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Sampling map",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            nu_min = np.nanmin([band.nu.min(), self.map.nu.min()])
            nu_max = np.nanmax([band.nu.max(), self.map.nu.max()])
            nus = [nu_min, *(self.map.nu[1:] + self.map.nu[:-1]) / 2, nu_max]

            # a fast separable approximation to the band integral
            power_map = 0
            for nu1, nu2, nu_bin_TRJ in zip(nus[:-1], nus[1:], self.map.data[0]):
                nu = np.linspace(nu1, nu2, 1024)  # in GHz
                tau = band.passband(nu)

                # in pW
                power_map += (
                    1e12 * band.efficiency * k_B * np.trapezoid(tau, x=1e9 * nu)
                ) * nu_bin_TRJ

            logger.debug(f"Computed power map for band {band.name}.")

            # nu is in GHz, f is in Hz
            nu_fwhm = beam.compute_angular_fwhm(
                fwhm_0=self.instrument.dets.primary_size.mean(),
                z=np.inf,
                nu=band.center,
            )

            nu_map_filter = beam.construct_beam_filter(
                fwhm=nu_fwhm,
                res=self.map.resolution,
            )

            filtered_power_map = beam.separably_filter_2d(power_map, nu_map_filter)

            logger.debug(f"Filtered power map for band {band.name}.")

            if len(self.map.t) > 1:
                map_power = sp.interpolate.RegularGridInterpolator(
                    (self.map.t, self.map.x_side, self.map.y_side),
                    filtered_power_map,
                    bounds_error=False,
                    fill_value=0,
                    method="linear",
                )((self.boresight.time, dx[band_mask], dy[band_mask]))

            else:
                map_power = sp.interpolate.RegularGridInterpolator(
                    (self.map.x_side, self.map.y_side),
                    filtered_power_map[0],
                    bounds_error=False,
                    fill_value=0,
                    method="linear",
                )((dx[band_mask], dy[band_mask]))

            if (map_power == 0).all():
                logger.warning("No power from map!")

            self.data["map"][band_mask] += map_power

            logger.debug(f"Computed map power for band {band.name}.")
