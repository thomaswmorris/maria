from __future__ import annotations

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

    def _interpolate_quantity(
        self, quantity, nu, pwv=None, base_temperature=None, elevation=45
    ):
        if pwv is None:
            pwv = np.median(self._side_zenith_pwv)
        if base_temperature is None:
            base_temperature = np.median(self._side_base_temperature)

        min_pwv = self._side_zenith_pwv.min()
        max_pwv = self._side_zenith_pwv.max()
        if (np.min(pwv) < min_pwv) or (np.max(pwv) > max_pwv):
            raise ValueError(f"PWV (in mm) must be between {min_pwv} and {max_pwv}.")

        min_elevation = self._side_elevation.min()
        max_elevation = self._side_elevation.max()
        if (np.min(elevation) < min_elevation) or (np.max(elevation) > max_elevation):
            raise ValueError(
                f"Elevation (in degrees) must be between {min_elevation} and {max_elevation}."
            )

        min_base_temp = self._side_base_temperature.min()
        max_base_temp = self._side_base_temperature.max()
        if (np.min(base_temperature) < min_base_temp) or (
            np.max(base_temperature) > max_base_temp
        ):
            raise ValueError(
                f"Base temperature (in Kelvin) must be between {min_base_temp:.01f} and {max_base_temp:.01f}."
            )

        return sp.interpolate.RegularGridInterpolator(
            points=(
                self._side_zenith_pwv,
                self._side_base_temperature,
                self._side_elevation,
                self._side_nu,
            ),
            values=getattr(self, f"_{quantity}"),
        )((pwv, base_temperature, elevation, nu))

    def emission(self, nu, pwv=None, base_temperature=None, elevation=45):
        return self._interpolate_quantity(
            "emission",
            nu=nu,
            pwv=pwv,
            base_temperature=base_temperature,
            elevation=elevation,
        )

    def opacity(self, nu, pwv=None, base_temperature=None, elevation=45):
        return self._interpolate_quantity(
            "opacity",
            nu=nu,
            pwv=pwv,
            base_temperature=base_temperature,
            elevation=elevation,
        )

    def path_delay(self, nu, pwv=None, base_temperature=None, elevation=45):
        return self._interpolate_quantity(
            "path_delay",
            nu=nu,
            pwv=pwv,
            base_temperature=base_temperature,
            elevation=elevation,
        )

    def transmission(self, nu, pwv=None, base_temperature=None, elevation=45):
        return np.exp(
            -self.opacity(
                nu,
                pwv=pwv,
                base_temperature=base_temperature,
                elevation=elevation,
            ),
        )
