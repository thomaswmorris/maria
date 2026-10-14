from __future__ import annotations

import logging
import os

import h5py
import numpy as np
import scipy as sp
from jax import scipy as jsp

from ..io import fetch
from ..site import REGIONS, InvalidRegionError, all_regions
from ..units import Quantity
from ..utils import is_numeric
from ..weather import Weather

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")


class AtmosphericSpectrum:
    @classmethod
    def from_weather(cls, weather: Weather, **kwargs):
        return cls(weather.region, weather.pressure_level, **kwargs)

    # def __init__(self, region, altitude: int = None, source: str = "am", refresh_cache=False):
    def __init__(self, region, pressure_level: float | Quantity = None, refresh_cache: bool = False):

        if region not in all_regions:
            raise InvalidRegionError(region)

        self.region = region

        # min_altitude = Quantity(REGIONS.loc[region].min_altitude, "m")
        # max_altitude = Quantity(REGIONS.loc[region].max_altitude, "m")

        if pressure_level is not None:
            self.pressure_level = Quantity(pressure_level, "hPa").pin("hPa")
        else:
            self.pressure_level = Weather(region).pressure_level

        # if (self.altitude < min_altitude) or (self.altitude > max_altitude):
        #     logger.warning(
        #         f"Supplied altitude {self.altitude} is outside the altitude range ({min_altitude} - {max_altitude}) "
        #         f"for region '{region}', extrapolated spectrum may be inaccurate"
        #     )

        self.cache_path = fetch(
            f"atmosphere/spectra/am/v4/{self.region}.h5",
            max_age=60 * 86400,
            refresh=refresh_cache,
        )

        with h5py.File(self.cache_path, "r") as f:
            # dims are (pressure, temperature, pwv, csc_el)
            self.side_pressure_level = Quantity(f["side_pressure_level"][:].astype(float), "hPa")
            self.side_effective_temperature = Quantity(f["side_effective_temperature"][:].astype(float), "K")
            self.side_pwv = Quantity(f["side_pwv"][:].astype(float), "mm")
            self.side_csc_el = f["side_csc_el"][:].astype(float)
            self.side_nu = Quantity(f["side_nu"][:].astype(float), "Hz")

            # set the min csc_el to 90.1 degrees
            # because sometimes rounded 32-bit types overrun 90 degrees in the interpolator
            # self.side_csc_el[0] = 0.999

            self.fields = {}
            for field in ["temperature_rayleigh_jeans", "opacity", "path_delay"]:
                self.fields[field] = sp.interpolate.interp1d(
                    self.side_pressure_level.hPa,
                    f[field][:],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                    axis=0,
                )(self.pressure_level.hPa)

    @property
    def nu_min(self):
        return Quantity(self.side_nu.min(), "Hz")

    @property
    def nu_max(self):
        return Quantity(self.side_nu.max(), "Hz")

    def __repr__(self):
        return f"""AtmosphericSpectrum({self.nu_min} - {self.nu_max}):
  region: {self.region}
  pressure_level: {self.pressure_level}"""

    def _interpolate_quantity(self, field, nu, pwv=None, effective_temperature=None, elevation=None):
        pwv = pwv if pwv is not None else np.median(self.side_pwv.mm)
        effective_temperature = (
            effective_temperature if effective_temperature is not None else np.median(self.side_effective_temperature.K)
        )
        elevation = elevation if elevation is not None else np.radians(45)

        min_pwv = self.side_pwv.mm.min()
        max_pwv = self.side_pwv.mm.max()
        if (np.min(pwv) < min_pwv) or (np.max(pwv) > max_pwv):
            raise ValueError(f"PWV (in mm) must be between {min_pwv} and {max_pwv}.")

        min_elevation = np.arcsin(1 / self.side_csc_el.max())
        max_elevation = np.arcsin(1 / self.side_csc_el.min())
        if (np.min(elevation) < min_elevation) or (np.max(elevation) > max_elevation):
            raise ValueError(f"Elevation (in radians) must be between {min_elevation} and {max_elevation}.")

        min_eff_temp = self.side_effective_temperature.K.min()
        max_eff_temp = self.side_effective_temperature.K.max()
        if (np.min(effective_temperature) < min_eff_temp) or (np.max(effective_temperature) > max_eff_temp):
            raise ValueError(f"Base temperature (in Kelvin) must be between {min_eff_temp:.01f} and {max_eff_temp:.01f}.")

        return jsp.interpolate.RegularGridInterpolator(
            points=self.points,
            values=self.fields[field],
        )((effective_temperature, pwv, 1 / np.sin(elevation), nu))

    @property
    def points(self):
        return (
            self.side_effective_temperature.K,
            self.side_pwv.mm,
            self.side_csc_el,
            self.side_nu.Hz,
        )

    def emission(self, nu, pwv=None, effective_temperature=None, elevation=None):
        return self._interpolate_quantity(
            "temperature_rayleigh_jeans",
            nu=nu,
            pwv=pwv,
            effective_temperature=effective_temperature,
            elevation=elevation,
        )

    def opacity(self, nu, pwv=None, effective_temperature=None, elevation=None):
        return self._interpolate_quantity(
            "opacity",
            nu=nu,
            pwv=pwv,
            effective_temperature=effective_temperature,
            elevation=elevation,
        )

    def path_delay(self, nu, pwv=None, effective_temperature=None, elevation=None):
        return self._interpolate_quantity(
            "path_delay",
            nu=nu,
            pwv=pwv,
            effective_temperature=effective_temperature,
            elevation=elevation,
        )

    def transmission(self, nu, pwv=None, effective_temperature=None, elevation=None):
        return np.exp(
            -self.opacity(
                nu,
                pwv=pwv,
                effective_temperature=effective_temperature,
                elevation=elevation,
            ),
        )
