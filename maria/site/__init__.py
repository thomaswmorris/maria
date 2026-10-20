from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from astropy.coordinates import EarthLocation

from ..io import read_yaml
from .region import REGIONS, InvalidRegionError, all_regions, plot_all_regions, region_table  # noqa
from .site import Site  # noqa

here, this_filename = os.path.split(__file__)

SITE_CONFIGS = {}
for sites_path in Path(f"{here}/sites").glob("*.yml"):
    SITE_CONFIGS.update(read_yaml(sites_path))

for site in SITE_CONFIGS:
    config = SITE_CONFIGS[site]
    SITE_CONFIGS[site]["instruments"] = ", ".join(config["instruments"]) if "instruments" in config else ""

SITE_DISPLAY_COLUMNS = [
    "region",
    "latitude",
    "longitude",
    "altitude",
    "instruments",
    "description",
]

site_table = pd.DataFrame(SITE_CONFIGS).T.sort_values("region")
site_table.loc[:, "timezone"] = region_table.loc[site_table.region, "timezone"].values
site_table = site_table.loc[:, SITE_DISPLAY_COLUMNS]
site_table.index.name = "name"
all_sites = list(site_table.index.values)


class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(
            f"The site '{invalid_site}' is not supported. Supported sites are:\n\n{site_table.to_string()}",
        )


def get_location(site_name):
    site = get_site(site_name)
    return EarthLocation.from_geodetic(
        lon=site.longitude.deg,
        lat=site.latitude.deg,
        height=site.altitude.m,
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
