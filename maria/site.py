import os
from dataclasses import dataclass, field

import pandas as pd
from astropy.coordinates import EarthLocation

from . import utils
from .weather import InvalidRegionError, supported_regions_table

here, this_filename = os.path.split(__file__)

SITE_CONFIGS = utils.io.read_yaml(f"{here}/configs/sites.yml")
SITE_PARAMS = set()
for key, config in SITE_CONFIGS.items():
    SITE_PARAMS |= set(config.keys())

DISPLAY_COLUMNS = ["site_description", "region", "latitude", "longitude", "altitude"]
site_data = pd.DataFrame(SITE_CONFIGS).T
all_sites = list(site_data.index.values)


class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(
            f"The site '{invalid_site}' is not supported. "
            f"Supported sites are:\n\n{site_data.loc[:, DISPLAY_COLUMNS].to_string()}"
        )


def get_location(site_name):
    site = get_site(site_name)
    return EarthLocation.from_geodetic(
        lon=site.longitude, lat=site.latitude, height=site.altitude
    )


def get_site_config(site_name="default", **kwargs):
    if site_name not in SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    SITE_CONFIG = SITE_CONFIGS[site_name].copy()
    for k, v in kwargs.items():
        SITE_CONFIG[k] = v
    return SITE_CONFIG


def get_site(site_name="default", **kwargs):
    return Site(**get_site_config(site_name, **kwargs))


@dataclass
class Site:
    site_description: str = ""
    region: str = "princeton"
    altitude: float = None  # in meters
    seasonal: bool = True
    diurnal: bool = True
    latitude: float = None  # in degrees
    longitude: float = None  # in degrees
    weather_quantiles: dict = field(default_factory=dict)
    site_documentation: str = ""

    """
    A class containing time-ordered pointing data. Pass a supported site (found at weathergen.sites),
    and a height correction if needed.
    """

    def __post_init__(self):
        if self.region not in supported_regions_table.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is None:
            self.longitude = supported_regions_table.loc[self.region].longitude

        if self.latitude is None:
            self.latitude = supported_regions_table.loc[self.region].latitude

        if self.altitude is None:
            self.altitude = supported_regions_table.loc[self.region].altitude

        self.earth_location = EarthLocation.from_geodetic(
            lon=self.longitude, lat=self.latitude, height=self.altitude
        )
