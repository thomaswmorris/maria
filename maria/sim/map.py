from __future__ import annotations

import logging
import os

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..beam import compute_angular_fwhm
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

        self.loading["map"] = da.zeros_like(dx, dtype=self.dtype)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Sampling map",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
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

                if len(self.map.t) > 1:
                    sample_T_RJ = sp.interpolate.RegularGridInterpolator(
                        (self.map.t, self.map.y_side, self.map.x_side),
                        smoothed_map.data[0, nu_index].compute(),
                        bounds_error=False,
                        fill_value=0,
                        method="linear",
                    )((self.boresight.t, dy[band_mask], dx[band_mask]))

                else:
                    sample_T_RJ = sp.interpolate.RegularGridInterpolator(
                        (self.map.y_side, self.map.x_side),
                        smoothed_map.data[0, nu_index, 0].compute(),
                        bounds_error=False,
                        fill_value=0,
                        method="linear",
                    )((dy[band_mask], dx[band_mask]))

                if (sample_T_RJ == 0).all():
                    logger.warning("No power from map!")

                self.loading["map"][band_mask] += 1e12 * k_B * sample_integral * sample_T_RJ

                logger.debug(f"Computed map power for band {band.name}.")

            if self.loading["map"][band_mask].sum().compute() == 0:
                logger.warning(f"No power from map for band {band}.")

                # things that are blackbodies that we can see are

                # filtered_band_power_map = (
                #     self.map.smooth(fwhm=band_fwhm).power(band).compute()
                # )
