from __future__ import annotations

import os

import h5py
import numpy as np
import scipy as sp
from jax import scipy as jsp

from ..io import fetch
from ..site import REGIONS, InvalidRegionError, all_regions
from ..units import Quantity

here, this_filename = os.path.split(__file__)


class AtmosphericSpectrum:
    def __init__(self, region, altitude: int = None, source: str = "am", refresh_cache=False):
        if region not in all_regions:
            raise InvalidRegionError(region)

        self.region = region
        self.altitude = altitude or REGIONS.loc[self.region, "altitude"]
        self.source = source

        self.cache_path = fetch(
            f"atmosphere/spectra/{self.source}/v2/{self.region}.h5",
            max_age=30 * 86400,
            refresh=refresh_cache,
        )

        key_mapping = {
            "rayleigh_jeans_temperature_K": "_emission",
            "opacity_nepers": "_opacity",
            "excess_path_m": "_path_delay",
        }

        with h5py.File(self.cache_path, "r") as f:
            # dims are (alt, temp, pwv, el)
            self.side_altitude = f["side_altitude_m"][:].astype(float)
            self.side_base_temperature = f["side_base_temperature_K"][:].astype(float)
            self.side_elevation = np.radians(f["side_elevation_deg"][:].astype(float))
            self.side_zenith_pwv = f["side_zenith_pwv_mm"][:].astype(float)
            self.side_nu = f["side_nu_Hz"][:].astype(float)

            for key, mapping in key_mapping.items():
                d = f[key]["relative"][:] * f[key]["scale"][:] + f[key]["offset"][:]

                setattr(
                    self,
                    mapping,
                    sp.interpolate.interp1d(self.side_altitude, d, axis=0)(self.altitude),
                )

    @property
    def nu_min(self):
        return Quantity(self.side_nu.min(), "Hz")

    @property
    def nu_max(self):
        return Quantity(self.side_nu.max(), "Hz")

    def __repr__(self):
        return f"""AtmosphericSpectrum({self.nu_min} - {self.nu_max}):
  region: {self.region}
  altitude: {Quantity(self.altitude, "m")}"""

    def _interpolate_quantity(self, quantity, nu, pwv=None, base_temperature=None, elevation=None):
        pwv = pwv or np.median(self.side_zenith_pwv)
        base_temperature = base_temperature or np.median(self.side_base_temperature)
        elevation = elevation or np.radians(45)

        min_pwv = self.side_zenith_pwv.min()
        max_pwv = self.side_zenith_pwv.max()
        if (np.min(pwv) < min_pwv) or (np.max(pwv) > max_pwv):
            raise ValueError(f"PWV (in mm) must be between {min_pwv} and {max_pwv}.")

        min_elevation = self.side_elevation.min()
        max_elevation = self.side_elevation.max()
        if (np.min(elevation) < min_elevation) or (np.max(elevation) > max_elevation):
            raise ValueError(f"Elevation (in degrees) must be between {min_elevation} and {max_elevation}.")

        min_base_temp = self.side_base_temperature.min()
        max_base_temp = self.side_base_temperature.max()
        if (np.min(base_temperature) < min_base_temp) or (np.max(base_temperature) > max_base_temp):
            raise ValueError(f"Base temperature (in Kelvin) must be between {min_base_temp:.01f} and {max_base_temp:.01f}.")

        return jsp.interpolate.RegularGridInterpolator(
            points=self.points,
            values=getattr(self, f"_{quantity}"),
        )((base_temperature, pwv, elevation, nu))

    @property
    def points(self):
        return (
            self.side_base_temperature,
            self.side_zenith_pwv,
            self.side_elevation,
            self.side_nu,
        )

    def emission(self, nu, pwv=None, base_temperature=None, elevation=None):
        return self._interpolate_quantity(
            "emission",
            nu=nu,
            pwv=pwv,
            base_temperature=base_temperature,
            elevation=elevation,
        )

    def opacity(self, nu, pwv=None, base_temperature=None, elevation=None):
        return self._interpolate_quantity(
            "opacity",
            nu=nu,
            pwv=pwv,
            base_temperature=base_temperature,
            elevation=elevation,
        )

    def path_delay(self, nu, pwv=None, base_temperature=None, elevation=None):
        return self._interpolate_quantity(
            "path_delay",
            nu=nu,
            pwv=pwv,
            base_temperature=base_temperature,
            elevation=elevation,
        )

    def transmission(self, nu, pwv=None, base_temperature=None, elevation=None):
        return np.exp(
            -self.opacity(
                nu,
                pwv=pwv,
                base_temperature=base_temperature,
                elevation=elevation,
            ),
        )
