import os
import warnings

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..instrument import beam
from ..units.constants import k_B

here, this_filename = os.path.split(__file__)


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
            1e-16 * np.random.standard_normal(size=dx.shape)
        )

        pbar = tqdm(self.instrument.bands, disable=not self.verbose)

        for band in pbar:
            pbar.set_description(f"Sampling map ({band.name})")

            band_mask = self.instrument.dets.band_name == band.name

            nu = np.linspace(band.nu_min, band.nu_max, 64)

            TRJ = sp.interpolate.interp1d(
                self.map.frequency,
                self.map.data,
                axis=0,
                kind="nearest",
                bounds_error=False,
                fill_value="extrapolate",
            )(nu)

            power_map = (
                1e12
                * k_B
                * np.trapezoid(
                    band.passband(nu)[:, None, None] * TRJ, axis=0, x=1e9 * nu
                )
            )

            # nu is in GHz, f is in Hz
            nu_fwhm = beam.compute_angular_fwhm(
                fwhm_0=self.instrument.dets.primary_size.mean(),
                z=np.inf,
                f=1e9 * band.center,
            )
            nu_map_filter = beam.construct_beam_filter(
                fwhm=nu_fwhm, res=self.map.resolution
            )
            filtered_power_map = beam.separably_filter(power_map, nu_map_filter)

            map_power = sp.interpolate.RegularGridInterpolator(
                (self.map.x_side, self.map.y_side),
                filtered_power_map,
                bounds_error=False,
                fill_value=0,
                method="linear",
            )((dx[band_mask], dy[band_mask]))

            if (map_power == 0).all():
                warnings.warn("No power from map!")

            self.data["map"][band_mask] += map_power