import os

import h5py
import numpy as np
import scipy as sp

from ..io import fetch
from ..site import InvalidRegionError, all_regions

here, this_filename = os.path.split(__file__)


class AtmosphericSpectrum:
    def __init__(self, region, source="am", refresh_cache=False):
        if region not in all_regions:
            raise InvalidRegionError(region)

        self.region = region
        self.source = source

        self.cache_path = fetch(
            f"atmosphere/spectra/{self.source}/{self.region}.h5",
            max_age=30 * 86400,
            refresh=refresh_cache,
        )

        key_mapping = {
            "brightness_temperature_rayleigh_jeans_K": "_emission",
            "opacity_nepers": "_opacity",
            "excess_path_m": "_path_delay",
        }

        with h5py.File(self.cache_path, "r") as f:
            self._side_nu = f["side_nu_GHz"][:]
            self._side_elevation = f["side_elevation_deg"][:]
            self._side_zenith_pwv = f["side_zenith_pwv_mm"][:]
            self._side_base_temperature = f["side_base_temperature_K"][:]

            for key, mapping in key_mapping.items():
                setattr(
                    self,
                    mapping,
                    f[key]["relative"][:] * f[key]["scale"][:] + f[key]["offset"][:],
                )

        self._transmission = np.exp(-self._opacity)
        del self._opacity

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
