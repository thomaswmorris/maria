import os
from dataclasses import dataclass, field

import pandas as pd
from astropy.coordinates import EarthLocation

from ..io import read_yaml
from ..utils import repr_lat_lon

here, this_filename = os.path.split(__file__)

SITE_CONFIGS = read_yaml(f"{here}/sites.yml")
SITE_PARAMS = set()
for key, config in SITE_CONFIGS.items():
    SITE_PARAMS |= set(config.keys())

SITE_DISPLAY_COLUMNS = [
    "description",
    "region",
    "latitude",
    "longitude",
    "altitude",
]
site_data = pd.DataFrame(SITE_CONFIGS).T
all_sites = list(site_data.index.values)

REGION_DISPLAY_COLUMNS = ["location", "country", "latitude", "longitude"]
supported_regions_table = pd.read_csv(f"{here}/regions.csv", index_col=0)
all_regions = list(supported_regions_table.index.values)


class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        super().__init__(
            f"The region '{invalid_region}' is not supported."
            f"Supported regions are:\n\n{supported_regions_table.loc[:, REGION_DISPLAY_COLUMNS].to_string()}"
        )


class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(
            f"""The site '{invalid_site}' is not supported."""
            f"""Supported sites are:\n\n{site_data.loc[:, SITE_DISPLAY_COLUMNS].to_string()}"""
        )


def get_location(site_name):
    site = get_site(site_name)
    return EarthLocation.from_geodetic(
        lon=site.longitude, lat=site.latitude, height=site.altitude
    )


def get_site_config(site_name="hoagie_haven", **kwargs):
    if site_name not in SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    SITE_CONFIG = SITE_CONFIGS[site_name].copy()
    for k, v in kwargs.items():
        SITE_CONFIG[k] = v
    return SITE_CONFIG


def get_site(site_name="hoagie_haven", **kwargs):
    return Site(**get_site_config(site_name=site_name, **kwargs))


@dataclass
class Site:
    description: str = ""
    region: str = "princeton"
    altitude: float = None  # in meters
    seasonal: bool = True
    diurnal: bool = True
    latitude: float = None  # in degrees
    longitude: float = None  # in degrees
    weather_quantiles: dict = field(default_factory=dict)
    documentation: str = ""

    """
    A class containing time-ordered pointing data. Pass a supported site (found at weathergen.sites),
    and a height correction if needed.
    """

    def __post_init__(self):
        if self.region not in supported_regions_table.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is None:
            self.longitude = float(supported_regions_table.loc[self.region].longitude)

        if self.latitude is None:
            self.latitude = float(supported_regions_table.loc[self.region].latitude)

        if self.altitude is None:
            self.altitude = float(supported_regions_table.loc[self.region].altitude)

        self.earth_location = EarthLocation.from_geodetic(
            lon=self.longitude, lat=self.latitude, height=self.altitude
        )

    def __repr__(self):
        parts = {
            "location": f"({repr_lat_lon(self.latitude, self.longitude)})",
            "region": self.region,
            "altitude": f"{self.altitude}m",
            "seasonal": self.seasonal,
            "diurnal": self.diurnal,
            "weather_quantiles": self.weather_quantiles,
        }

        return rf"Site({', '.join([f'{k}={v}' for k, v in parts.items()])})"
