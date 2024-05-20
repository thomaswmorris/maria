import os

import h5py
import numpy as np
import scipy as sp

from ..io import fetch_cache
from ..site import InvalidRegionError, all_regions

here, this_filename = os.path.split(__file__)

SPECTRA_CACHE_BASE = "/tmp/maria-data/atmosphere/spectra"
SPECTRA_SOURCE_BASE = (
    "https://github.com/thomaswmorris/maria-data/raw/master/atmosphere/spectra"  # noqa
)


class Spectrum:
    def __init__(self, region, source="am", refresh_cache=False):
        if region not in all_regions:
            raise InvalidRegionError(region)

        self.region = region
        self.source = source

        self.source_url = f"{SPECTRA_SOURCE_BASE}/{source}/{self.region}.h5"  # noqa
        self.cache_path = f"{SPECTRA_CACHE_BASE}/{source}/{self.region}.h5"  # noqa

        fetch_cache(
            source_url=self.source_url,
            cache_path=self.cache_path,
            max_age=30 * 86400,
            refresh=refresh_cache,
        )

        with h5py.File(self.cache_path, "r") as f:
            self._side_nu = f["side_nu_GHz"][:]
            self._side_elevation = f["side_elevation_deg"][:]
            self._side_zenith_pwv = f["side_zenith_pwv_mm"][:]
            self._side_base_temperature = f["side_base_temperature_K"][:]

            self._emission = f["emission_temperature_rayleigh_jeans_K"][:]
            self._transmission = np.exp(-f["opacity_nepers"][:])
            self._excess_path = 1e6 * (
                f["excess_path"][:] + f["offset_excess_path_m"][:]
            )

    def emission(self, nu, zenith_pwv=None, base_temperature=None, elevation=45):
        if zenith_pwv is None:
            zenith_pwv = np.median(self._side_zenith_pwv)
        if base_temperature is None:
            base_temperature = np.median(self._side_base_temperature)

        return sp.interpolate.RegularGridInterpolator(
            points=(
                self._side_zenith_pwv,
                self._side_base_temperature,
                self._side_elevation,
                self._side_nu,
            ),
            values=self._emission,
        )((zenith_pwv, base_temperature, elevation, nu))

    def transmission(self, nu, zenith_pwv=None, base_temperature=None, elevation=45):
        if zenith_pwv is None:
            zenith_pwv = np.median(self._side_zenith_pwv)
        if base_temperature is None:
            base_temperature = np.median(self._side_base_temperature)

        return sp.interpolate.RegularGridInterpolator(
            points=(
                self._side_zenith_pwv,
                self._side_base_temperature,
                self._side_elevation,
                self._side_nu,
            ),
            values=self._transmission,
        )((zenith_pwv, base_temperature, elevation, nu))
