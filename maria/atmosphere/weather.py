import os
from datetime import datetime

import h5py
import numpy as np
import pytz
import scipy as sp

from ..constants import g
from ..io import fetch
from ..site import InvalidRegionError, all_regions, supported_regions_table
from ..utils import get_utc_day_hour, get_utc_year_day

here, this_filename = os.path.split(__file__)

WEATHER_CACHE_BASE = "/tmp/maria-data/weather"
WEATHER_SOURCE_BASE = "https://github.com/thomaswmorris/maria-data/raw/master/atmosphere/weather"  # noqa F401


def get_vapor_pressure(air_temp, rel_hum):  # units are (°K, %)
    T = air_temp - 273.15  # in °C
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    gamma = np.log(1e-2 * rel_hum) + b * T / (c + T)
    return a * np.exp(gamma)


def get_saturation_pressure(air_temp):  # units are (°K, %)
    T = air_temp - 273.15  # in °C
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    return a * np.exp(b * T / (c + T))


def get_dew_point(air_temp, rel_hum):  # units are (°K, %)
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    p_vap = get_vapor_pressure(air_temp, rel_hum)
    return c * np.log(p_vap / a) / (b - np.log(p_vap / a)) + 273.15


def get_relative_humidity(air_temp, dew_point):
    T, DP = air_temp - 273.15, dew_point - 273.15  # in °C
    b, c = 17.67, 238.88
    return 1e2 * np.exp(b * DP / (c + DP) - b * T / (c + T))


def relative_to_absolute_humidity(air_temp, rel_hum):
    return 1e-2 * rel_hum * get_saturation_pressure(air_temp) / (461.5 * air_temp)


def absolute_to_relative_humidity(air_temp, abs_hum):
    return 1e2 * 461.5 * air_temp * abs_hum / get_saturation_pressure(air_temp)


class Weather:
    def __init__(
        self,
        region: str = "chajnantor",
        t: float = 0,
        altitude: float = None,
        utc_time: str = "",
        local_time: str = "",
        time_zone: str = "",
        quantiles: dict = {},
        override: dict = {},
        source: str = "era5",
        refresh_cache: bool = False,
    ):
        if region not in all_regions:
            raise InvalidRegionError(self.region)

        self.region = region
        self.t = t
        self.altitude = altitude
        self.utc_time = utc_time
        self.local_time = local_time
        self.time_zone = time_zone
        self.quantiles = quantiles
        self.override = override
        self.source = source

        self.cache_path = fetch(
            f"atmosphere/weather/{source}/{self.region}.h5",
            max_age=30 * 86400,
            refresh=refresh_cache,
        )

        if self.altitude is None:
            self.altitude = supported_regions_table.loc[self.region, "altitude"]

        self.t = np.round(self.t, 0)
        self.altitude = np.round(self.altitude, 0)
        self.time_zone = supported_regions_table.loc[self.region, "timezone"]
        self.utc_datetime = datetime.fromtimestamp(self.t).astimezone(pytz.utc)
        self.utc_time = self.utc_datetime.ctime()
        self.local_time = self.utc_datetime.astimezone(
            pytz.timezone(self.time_zone)
        ).ctime()

        with h5py.File(self.cache_path, "r") as f:
            self.utc_day_hour = get_utc_day_hour(self.t)
            self.utc_year_day = get_utc_year_day(self.t)

            self.quantile_levels = f["quantile_levels"][:]
            self.pressure_levels = f["pressure_levels"][:]

            self.fields = list(f["data"].keys())

            # for fun sampling things
            year_day_side = f["year_day_side"][:]
            day_hour_side = f["day_hour_side"][:]
            year_day_edge_index = f["year_day_edge_index"][:]
            day_hour_edge_index = f["day_hour_edge_index"][:]

            YEAR_DAY_EDGE_INDEX, DAY_HOUR_EDGE_INDEX = np.meshgrid(
                year_day_edge_index, day_hour_edge_index, indexing="ij"
            )

            for attr in self.fields:
                quantiles = (
                    f["data"][attr]["normalized_quantiles"][:]
                    * f["data"][attr]["scale"][()]
                    + f["data"][attr]["mean"][()]
                )
                y = sp.interpolate.RegularGridInterpolator(
                    (year_day_side, day_hour_side, self.quantile_levels),
                    quantiles[YEAR_DAY_EDGE_INDEX, DAY_HOUR_EDGE_INDEX],
                )((self.utc_year_day, self.utc_day_hour, self.quantiles.get(attr, 0.5)))
                setattr(self, attr, y)

    @property
    def absolute_humidity(self):
        return relative_to_absolute_humidity(self.temperature, self.humidity)

    @property
    def dew_point(self):
        return get_dew_point(self.temperature, self.humidity)

    @property
    def wind_bearing(self):
        return np.arctan2(-self.wind_east, self.wind_north) % (2 * np.pi)

    @property
    def wind_speed(self):
        return np.sqrt(self.wind_east**2 + self.wind_north**2)

    @property
    def altitude_levels(self):
        return self.geopotential / g

    @property
    def pwv(self):
        if "pwv" in self.override.keys():
            return self.override["pwv"]
        z = np.linspace(self.altitude * g, self.geopotential.max(), 1024)
        return np.trapezoid(
            np.interp(z, self.geopotential, self.absolute_humidity), z / g
        )
