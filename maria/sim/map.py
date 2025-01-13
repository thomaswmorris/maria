from __future__ import annotations

import logging
import os

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..instrument import beam

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

        self.data["map"] = da.zeros_like(dx, dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Sampling map",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_mask = self.instrument.dets.band_name == band.name

            band_fwhm = beam.compute_angular_fwhm(
                fwhm_0=self.instrument.dets.primary_size.mean(),
                z=np.inf,
                nu=band.center,
            )

            filtered_band_power_map = (
                self.map.smooth(fwhm=band_fwhm).power(band).compute()
            )

            logger.debug(f"Computed power map for band {band.name}.")

            if len(self.map.t) > 1:
                map_power = sp.interpolate.RegularGridInterpolator(
                    (self.map.t, self.map.x_side, self.map.y_side),
                    filtered_band_power_map,
                    bounds_error=False,
                    fill_value=0,
                    method="linear",
                )((self.boresight.t, dx[band_mask], dy[band_mask]))

            else:
                map_power = sp.interpolate.RegularGridInterpolator(
                    (self.map.x_side, self.map.y_side),
                    filtered_band_power_map[0],
                    bounds_error=False,
                    fill_value=0,
                    method="linear",
                )((dx[band_mask], dy[band_mask]))

            if (map_power == 0).all():
                logger.warning("No power from map!")

            self.data["map"][band_mask] += map_power

            logger.debug(f"Computed map power for band {band.name}.")
