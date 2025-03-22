from __future__ import annotations

import os

import arrow
import h5py
import numpy as np
import pandas as pd
import scipy as sp

from ..constants import g
from ..io import DEFAULT_TIME_FORMAT, fetch
from ..site import REGIONS, InvalidRegionError, all_regions
from ..units import Quantity
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
        time: arrow.Arrow = None,
        altitude: float = None,
        quantiles: dict = {},
        override: dict = {},
        source: str = "era5",
        refresh_cache: bool = False,
    ):
        if region not in all_regions:
            raise InvalidRegionError(self.region)

        self.region = region
        self.base_altitude = altitude
        self.quantiles = quantiles
        self.override = override
        self.source = source

        time = time if time is not None else arrow.now().to("utc")
        self.time = arrow.get(time)

        self.cache_path = fetch(
            f"atmosphere/weather/{source}/{self.region}.h5",
            max_age=30 * 86400,
            refresh=refresh_cache,
        )

        if self.base_altitude is None:
            self.base_altitude = REGIONS.loc[self.region, "altitude"]

        self.base_altitude = np.round(self.base_altitude, 0)
        self.time_zone = REGIONS.loc[self.region, "timezone"]
        self.local_time = self.time.to(self.time_zone)

        self.utc_day_hour = get_utc_day_hour(self.time.timestamp())
        self.utc_year_day = get_utc_year_day(self.time.timestamp())

        self.data = {}

        self.region_quantiles = {}

        with h5py.File(self.cache_path, "r") as f:
            self.quantile_levels = f["quantile_levels"][:]
            self.pressure_levels = f["pressure_levels"][:]

            self.fields = list(f["data"].keys())

            # for fun sampling things
            year_day_side = f["year_day_side"][:]
            day_hour_side = f["day_hour_side"][:]
            year_day_edge_index = f["year_day_edge_index"][:]
            day_hour_edge_index = f["day_hour_edge_index"][:]

            YEAR_DAY_EDGE_INDEX, DAY_HOUR_EDGE_INDEX = np.meshgrid(
                year_day_edge_index,
                day_hour_edge_index,
                indexing="ij",
            )

            for attr in f["data"].keys():
                self.region_quantiles[attr] = (
                    f["data"][attr]["normalized_quantiles"][:] * f["data"][attr]["scale"][()] + f["data"][attr]["mean"][()]
                )

            # quantiles["wind_speed"] = np.sqrt(quantiles["wind_east"]**2 + quantiles["wind_north"]**2)

            for attr, data in self.region_quantiles.items():
                y = sp.interpolate.RegularGridInterpolator(
                    (year_day_side, day_hour_side, self.quantile_levels),
                    data[YEAR_DAY_EDGE_INDEX, DAY_HOUR_EDGE_INDEX],
                )((self.utc_year_day, self.utc_day_hour, self.quantiles.get(attr, 0.5)))
                self.data[attr] = y

        wind_speed_correction_factor = self.wind_speed / np.sqrt(
            self.wind_east**2 + self.wind_north**2,
        )
        self.wind_north *= wind_speed_correction_factor
        self.wind_east *= wind_speed_correction_factor

        self.altitude = self.geopotential / g

        self.is_scalar = False

        # handle the overrided parameters
        if "pwv" in self.override.keys():
            force_absolute_humidity = self.override["pwv"] / self.pwv * self.absolute_humidity
            self.humidity = absolute_to_relative_humidity(self.temperature, force_absolute_humidity)

    def __getattr__(self, attr):
        if attr in self.data:
            return self.data[attr]

        raise AttributeError()

    @property
    def absolute_humidity(self):
        return relative_to_absolute_humidity(self.temperature, self.humidity)

    @property
    def dew_point(self):
        return get_dew_point(self.temperature, self.humidity)

    @property
    def wind_bearing(self):
        return np.arctan2(-self.wind_east, self.wind_north) % (2 * np.pi)

    # @property
    # def wind_speed(self):
    #     return np.sqrt(self.wind_east**2 + self.wind_north**2 + self.wind_vertical**2)

    def layers(self):
        df = pd.DataFrame(self.data)
        df.insert(0, "altitude", df.geopotential.values / g)
        df.insert(1, "pressure", 1e2 * self.pressure_levels)

        df = df.loc[df.altitude > self.base_altitude]
        df.index = np.arange(len(df))

        absolute_humidity = relative_to_absolute_humidity(df.temperature, df.humidity)

        df.insert(2, "h_bottom", 0.0)
        df.insert(3, "h_top", 0.0)
        df.insert(4, "h_thickness", 0.0)
        df.insert(5, "total_water", 0.0)
        df.insert(6, "absolute_humidity", absolute_humidity)

        h_values = df.altitude.values
        h_bins = np.array(
            [
                self.base_altitude,
                *(h_values[:-1] + h_values[1:]) / 2,
                h_values[-1] + 100,
            ]
        )

        for layer_index, (h1, h2) in enumerate(zip(h_bins[:-1], h_bins[1:])):
            h = np.linspace(h1, h2, 256)

            absolute_humidity = sp.interpolate.interp1d(
                self.altitude,
                self.absolute_humidity,
                bounds_error=False,
                fill_value="extrapolate",
            )(h)

            df.loc[layer_index, ["h_bottom", "h_top", "h_thickness"]] = h1, h2, h2 - h1
            df.loc[layer_index, "total_water"] = np.trapezoid(absolute_humidity, x=h)

        df.loc[:, "altitude"] = (h_bins[:-1] + h_bins[1:]) / 2

        return df

    @property
    def pwv(self):
        return self.layers().total_water.sum()

    def __call__(self, altitude):
        res = {}
        for field in [*self.fields, "absolute_humidity"]:
            res[field] = sp.interpolate.interp1d(self.altitude, getattr(self, field))(
                altitude,
            )
        return res

    def __repr__(self):
        parts = []
        parts.append(f"region={self.region}")
        parts.append(f"time={self.local_time.format('MMM D HH:mm:ss ZZ')}")
        parts.append(f"altitude={Quantity(self.base_altitude, 'm')}")
        parts.append(f"pwv={Quantity(self.pwv, 'mm')}")
        return f"Weather({', '.join(parts)})"
