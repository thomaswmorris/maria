from __future__ import annotations

import os

import arrow
import logging
import subprocess
import numpy as np
import pandas as pd
import scipy as sp

from ..io import fetch, read_weather_quantile_data
from ..site import REGIONS, InvalidRegionError, all_regions
from ..units import Quantity
from ..utils import get_utc_day_hour, get_utc_year_day, relative_to_absolute_humidity, absolute_to_relative_humidity, dew_point, compute_air_density, vapor_pressure

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

WEATHER_CACHE_BASE = "/tmp/maria-data/weather"
WEATHER_SOURCE_BASE = "https://github.com/thomaswmorris/maria-data/raw/master/atmosphere/weather"  # noqa F401


g = Quantity(9.81, "m s^-2")
water_density = Quantity(1e3, "kg m^-3")

class Weather:
    def __init__(
        self,
        region: str = "chajnantor",
        time: arrow.Arrow = None,
        altitude: float = None,
        pressure_level: float = None,
        diurnal: bool = True,
        seasonal: bool = True,
        quantiles: dict = {},
        override: dict = {},
        source: str = "era5",
        refresh_cache: bool = False,
    ):

        if region not in all_regions:
            raise InvalidRegionError(region)
            
        self.region = region
        self.quantiles = quantiles
        self.override = override
        self.source = source

        time = time if time is not None else arrow.now().to("utc")
        self.time = arrow.get(time)

        self.cache_path = fetch(
            f"atmosphere/weather/{source}/v2/{self.region}.h5",
            refresh=refresh_cache,
        )

        self.timezone = REGIONS.loc[self.region, "timezone"]
        self.local_time = self.time.to(self.timezone)

        self.utc_year_day = get_utc_year_day(self.time.timestamp())
        self.utc_day_hour = get_utc_day_hour(self.time.timestamp())

        self.qdata = read_weather_quantile_data(self.cache_path)

        n_year_day_side = len(self.qdata["metadata"]["side_year_day"])
        n_day_hour_side = len(self.qdata["metadata"]["side_day_hour"])

        year_day_wrap_index = np.arange(-1, n_year_day_side + 1) % n_year_day_side
        day_hour_wrap_index = np.arange(-1, n_day_hour_side + 1) % n_day_hour_side
        year_day_wrapped_values = [
            self.qdata["metadata"]["side_year_day"][-1] - 365,
            *self.qdata["metadata"]["side_year_day"],
            self.qdata["metadata"]["side_year_day"][0] + 365,
        ]

        day_hour_wrapped_values = [
            self.qdata["metadata"]["side_day_hour"][-1] - 24,
            *self.qdata["metadata"]["side_day_hour"],
            self.qdata["metadata"]["side_day_hour"][0] + 24,
        ]

        self.data = {"levels": {}}
        for field in self.qdata["levels"].keys():

            field_data = self.qdata["levels"][field].copy()

            # interpolate by quantile
            field_data = sp.interpolate.interp1d(self.qdata["metadata"]["side_quantile"][:], 
                                                 field_data, axis=0)(quantiles.get(field, 0.5))

            # interpolate by year day
            if seasonal:
                field_data = sp.interpolate.interp1d(year_day_wrapped_values, 
                                                     np.take(field_data, indices=year_day_wrap_index, axis=0),
                                                     axis=0,
                                                     )(self.utc_year_day)
            else:
                field_data = np.median(field_data, axis=0)

            # interpolate by year day
            if diurnal:
                field_data = sp.interpolate.interp1d(day_hour_wrapped_values, 
                                                     np.take(field_data, indices=day_hour_wrap_index, axis=0),
                                                     axis=0,
                                                     )(self.utc_day_hour)
            else:
                field_data = np.median(field_data, axis=0)

            self.data["levels"][field] = Quantity(field_data, self.qdata["metadata"]["units"][field])

        # do some adjustments
        wind_factor = self.wind_speed / (self.wind_east**2 + self.wind_north**2)**0.5
        self.data["levels"]["wind_east"] *= wind_factor
        self.data["levels"]["wind_north"] *= wind_factor

        if altitude is None:
            if pressure_level is None:
                self.altitude = Quantity(REGIONS.loc[region, "altitude"], "m")
            else:
                self.pressure_level = Quantity(pressure_level, "hPa")
        else:
            self.altitude = Quantity(altitude, "m")
            if pressure_level is not None:
                logger.warning("Ignoring argument 'pressure_level'")


        if "pwv" in self.override:
            self.water_factor = Quantity(self.override["pwv"], "mm") / self.pwv
            new_water_vapor = self.water_factor * self.water_vapor.to("kg m^-3")
            self.data["levels"]["humidity"] = absolute_to_relative_humidity(temperature=self.temperature.K,
                                                                            abs_hum=new_water_vapor)

    @property
    def z(self):
        return self.data["levels"]["geopotential"] / g
    
    @property
    def altitude(self):
        return self._altitude.pin("m")

    @altitude.setter
    def altitude(self, value):
        self._altitude = Quantity(value, "meters").pin("m")

        
        self._pressure_level = Quantity(np.exp(sp.interpolate.interp1d(self.z.m, np.log(self.pressure.Pa), bounds_error=False, fill_value="extrapolate")(self._altitude.m)), "Pa")

        if self._altitude.m > 50000:
            logger.warning("Extrapolated weather parameters may be inaccurate for altitudes greater than 50 km")

    @property
    def pressure_level(self):
        return self._pressure_level.pin("hPa")

    @pressure_level.setter
    def pressure_level(self, value):
        self._pressure_level = Quantity(value, "hPa")
        self._altitude = Quantity(sp.interpolate.interp1d(np.log(self.pressure.Pa), self.z.m, bounds_error=False, fill_value="extrapolate")(np.log(self._pressure_level.Pa)), "m")

        if self._pressure_level.hPa > 1100 or self._pressure_level.hPa < 1:
            logger.warning("Extrapolated weather parameters may be inaccurate pressure levels greater than 1100 hPa or less than 1 hPa.")

    @property
    def pressure(self):
        return Quantity(self.qdata["metadata"]["side_pressure"], units="Pa")

    @property
    def water_vapor(self):
        return Quantity(relative_to_absolute_humidity(humidity=self.humidity,
                                                      temperature=self.temperature.K), units="kg m^-3")
    


    def z_samples(self, dz: float = 1e1):
        return np.arange(self.altitude.m, self.z.m.max(), 1e1)

    @property
    def pwv(self):
        z_samples = self.z_samples(dz=10)
        values = self(altitude=z_samples, fields=["water_vapor"])
        return Quantity(np.trapezoid(values["water_vapor"] / water_density, x=z_samples), "m")

    @property
    def effective_temperature(self):
        z_samples = self.z_samples(dz=10)
        values = self(altitude=z_samples, fields=["air_density", "temperature"])
        return (values["temperature"] * values["air_density"]).sum() / values["air_density"].sum()
        
    @property
    def dew_point(self):
        return Quantity(dew_point(temperature=self.temperature.K, humidity=self.humidity), "K")

    @property
    def vapor_pressure(self):
        return Quantity(vapor_pressure(temperature=self.temperature.K, humidity=self.humidity), "Pa")

    @property
    def wind_bearing(self):
        return np.arctan2(-self.wind_east.to("m/s"), self.wind_north.to("m/s")) % (2 * np.pi)

    @property
    def air_density(self):
        return Quantity(compute_air_density(pressure=self.pressure.Pa, 
                                    temperature=self.temperature.K, 
                                    humidity=self.humidity), "kg m^-3")

    @property
    def base_pressure_index(self):
        return np.where(self.pressure >= self.pressure_level)[0][-1]
    
    def __getattr__(self, attr):
        for kind in ["levels"]:
            if attr in self.data[kind]:
                return self.data[kind][attr]

        raise AttributeError()

    def layers(self):

        df = pd.DataFrame(index=np.arange(len(self.qdata["metadata"]["side_pressure"])))

        for field in [*list(self.data["levels"].keys())]:

            df.loc[:, field] = [repr(v) for v in getattr(self, field)]

        return df

    def compute_base_global_quantiles(self, fields = None):

        if fields is None:
            fields = self.data["levels"].keys()

        base_global_quantiles = {}

        base_pressure_index = self.base_pressure_index

        for field in fields:
            
            base_field_value = self.data["levels"][field][base_pressure_index].to(self.qdata["metadata"]["units"][field])
            base_global_quantile_values = np.median(self.qdata["levels"][field][..., base_pressure_index], axis=(-2, -1))

            base_global_quantiles[field] = sp.interpolate.interp1d(base_global_quantile_values, 
                                                                self.qdata["metadata"]["side_quantile"])(base_field_value)

        return base_global_quantiles


    def am_config(self, nu_min: float = 1e9, nu_max: float = 1e12, nu_step: float = 1e9, el: float = 45):

        nu_min = Quantity(nu_min, "Hz")
        nu_max = Quantity(nu_max, "Hz")
        nu_step = Quantity(nu_step, "Hz")
        el = Quantity(el, "deg")
        
        config_header = f"""f {nu_min.Hz} Hz {nu_max.Hz} Hz {nu_step.Hz} Hz
output f Hz Trj K tau neper L m
tol 0
za {90 - el.deg} deg
T0 0 K
"""
        pressure_levels = [self.pressure_level, *self.pressure[self.pressure < self.pressure_level]][::-1]
        
        layer_data = self(pressure_level=pressure_levels)
        pwv_derivative = 0.5 * (layer_data["water_vapor"][:-1] + layer_data["water_vapor"][1:])
        layer_data["pwv"] = Quantity([0, *pwv_derivative / water_density * np.diff(layer_data["z"].m)], "m")
        if layer_data["pwv"].sum() > 0:
            layer_data["pwv"] *= self.pwv / layer_data["pwv"].sum()
        
        config_parts = [config_header]
        for level_index, pressure_level in enumerate(pressure_levels):

            layer_config = f"""layer
Pbase {layer_data["pressure"][level_index].Pa:.03f} Pa
Tbase {layer_data["temperature"][level_index].K:.03f} K
column h2o {np.abs(layer_data["pwv"][level_index].um)} um_pwv
column o3 vmr {(28.96 / 48) * layer_data["ozone"][level_index].to("kg m^-3")}
column dry_air vmr"""

            config_parts.append(layer_config)

        return "\n\n".join(config_parts)

    def compute_am_spectrum(self,
                            am_path: str,
                            nu_min: Quantity | float = 1e9,
                            nu_max: Quantity | float = 1e12,
                            nu_step: Quantity | float = 1e9,
                            el: str | Quantity | float = 45,
                            config_path: str = "/tmp/config.amc"):

        c = self.am_config(nu_min=nu_min, nu_max=nu_max, nu_step=nu_step, el=el)
        
        with open(config_path, "w") as f:
            f.write(c)

        proc = subprocess.run([am_path, config_path], capture_output=True, text=True)
        spec_values = np.array(proc.stdout.split()).reshape(-1, 4).astype(float)

        return {
            "nu": Quantity(spec_values[:, 0], "Hz"), 
            "temperature_rayleigh_jeans": Quantity(spec_values[:, 1], "K_RJ"),
            "opacity": spec_values[:, 2],
            "path_delay": Quantity(spec_values[:, 3], "m"),
            "stderr": proc.stderr,
        }


    def __call__(self, altitude=None, pressure_level=None, fields: list = None):

        if fields is None:
            fields = ["z", "pressure", *list(self.data["levels"].keys()), "water_vapor", "air_density", "dew_point"]

        if pressure_level:
            altitude = sp.interpolate.interp1d(self.pressure.Pa, self.z.m, bounds_error=False, fill_value="extrapolate")(Quantity(pressure_level, "hPa").Pa)

        altitude = Quantity(altitude, "m").m

        res = {}
        for field in fields:
            
            values = getattr(self, field)
            if isinstance(values, Quantity):
                res[field] = Quantity(sp.interpolate.interp1d(self.z.m, values.human_value, bounds_error=False, fill_value="extrapolate")(altitude), values.human_units)
            else:
                res[field] = sp.interpolate.interp1d(self.z.m, values)(altitude)

        return res

    def __repr__(self):
        return f"""Weather:
  region: {self.region}
  time: {self.local_time.format('MMM D HH:mm:ss ZZ')} ({self.timezone})
  altitude: {self.altitude}
  pressure_level: {self.pressure_level}
  base_temperature: {self(altitude=self.altitude, fields=["temperature"])["temperature"]}
  effective_temperature: {self.effective_temperature}
  pwv: {self.pwv}"""
    