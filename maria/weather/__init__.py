import os
from dataclasses import dataclass, field
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytz
import requests
import scipy as sp

from ..utils import get_utc_day_hour, get_utc_year_day
from ..utils.constants import g
from ..utils.weather import get_dew_point, relative_to_absolute_humidity

here, this_filename = os.path.split(__file__)

DISPLAY_COLUMNS = ["location", "country", "latitude", "longitude"]
supported_regions_table = pd.read_csv(f"{here}/regions.csv", index_col=0)
all_regions = list(supported_regions_table.index.values)

WEATHER_DATA_URL_BASE = (
    "https://github.com/thomaswmorris/maria/raw/master/maria/atmosphere/spectra"
)


class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        super().__init__(
            f"The region '{invalid_region}' is not supported."
            f"Supported regions are:\n\n{supported_regions_table.loc[:, DISPLAY_COLUMNS].to_string()}"
        )


@dataclass
class Weather:
    t: float = 0
    region: str = "chajnantor"
    altitude: float = None
    utc_time: str = ""
    local_time: str = ""
    time_zone: str = ""
    quantiles: dict = field(default_factory=dict)
    override: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.region not in all_regions:
            raise InvalidRegionError(self.region)

        self._weather_path = f"{here}/data/{self.region}.h5"

        # download the data as needed
        if not os.path.exists(self._weather_path):
            print("getting spectrum data...")
            url = f"{WEATHER_DATA_URL_BASE}/{self.region}.h5"
            r = requests.get(url)
            with open(self._weather_path, "wb") as f:
                f.write(r.content)

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

        with h5py.File(self._weather_path, "r") as f:
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
        return np.trapz(np.interp(z, self.geopotential, self.absolute_humidity), z / g)
