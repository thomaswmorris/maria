from __future__ import annotations

import os
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation
from matplotlib import pyplot as plt

from ..io import fetch, read_yaml
from ..utils import repr_lat_lon

here, this_filename = os.path.split(__file__)

SITE_CONFIGS = {}
for sites_path in Path(f"{here}/sites").glob("*.yml"):
    SITE_CONFIGS.update(read_yaml(sites_path))

for site in SITE_CONFIGS:
    config = SITE_CONFIGS[site]
    SITE_CONFIGS[site]["instruments"] = (
        ", ".join(config["instruments"]) if "instruments" in config else ""
    )

SITE_DISPLAY_COLUMNS = [
    "description",
    "instruments",
    "region",
    "latitude",
    "longitude",
    "altitude",
]
site_data = pd.DataFrame(SITE_CONFIGS).T.sort_values("region")
all_sites = list(site_data.index.values)

REGION_DISPLAY_COLUMNS = ["location", "country", "latitude", "longitude"]
supported_regions_table = pd.read_csv(f"{here}/regions.csv", index_col=0)
all_regions = list(supported_regions_table.index.values)

height_map = hp.fitsfunc.read_map(fetch("world_heightmap.fits")).astype(np.uint16)
height_map = 32 * np.where(height_map < 255, height_map, np.nan)


class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        super().__init__(
            f"The region '{invalid_region}' is not supported. "
            f"Supported regions are:\n\n{supported_regions_table.loc[:, REGION_DISPLAY_COLUMNS].to_string()}",
        )


class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(
            f"The site '{invalid_site}' is not supported. "
            f"Supported sites are:\n\n{site_data.loc[:, SITE_DISPLAY_COLUMNS].to_string()}",
        )


def get_location(site_name):
    site = get_site(site_name)
    return EarthLocation.from_geodetic(
        lon=site.longitude,
        lat=site.latitude,
        height=site.altitude,
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


class Site:
    """
    An object representing a point on the earth that parametrizes observing conditions.
    """

    def __init__(
        self,
        description: str = "",
        region: str = "princeton",
        altitude: float = None,  # in meters
        seasonal: bool = True,
        diurnal: bool = True,
        latitude: float = None,  # in degrees
        longitude: float = None,  # in degrees
        weather_quantiles: dict = {},
        documentation: str = "",
        instruments: list = [],
    ):
        self.description = description
        self.region = region
        self.altitude = altitude
        self.seasonal = seasonal
        self.diurnal = diurnal
        self.latitude = latitude
        self.longitude = longitude
        self.weather_quantiles = weather_quantiles
        self.documentation = documentation
        self.instruments = instruments

        if self.region not in supported_regions_table.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is None:
            self.longitude = float(supported_regions_table.loc[self.region].longitude)

        if self.latitude is None:
            self.latitude = float(supported_regions_table.loc[self.region].latitude)

        if self.altitude is None:
            self.altitude = float(supported_regions_table.loc[self.region].altitude)

        self.earth_location = EarthLocation.from_geodetic(
            lon=self.longitude,
            lat=self.latitude,
            height=self.altitude,
        )

    def plot(self, res=0.025):

        kwargs = {
            "rot": (self.longitude, self.latitude),
            "cmap": "gist_earth",
            "flip": "geo",
            "badcolor": "gray",
            "reso": 60 * res,
            "return_projected_map": True,
            "no_plot": True,
        }

        zoom_map = hp.gnomview(
            height_map,
            xsize=8 / res,
            **kwargs,
        )  # , norm=mpl.colors.Normalize(vmin=0, vmax=5e3))
        wide_map = hp.gnomview(
            height_map,
            xsize=90 / res,
            **kwargs,
        )  # , norm=mpl.colors.Normalize(vmin=0, vmax=5e3))

        fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)

        zoom_vmax = np.nanpercentile(zoom_map[zoom_map > 0], q=99)
        wide_vmax = np.nanpercentile(wide_map[wide_map > 0], q=99)

        cmap = mpl.cm.gist_ncar
        cmap.set_bad("gray", 1.0)

        axes[0].imshow(
            wide_map[::-1],
            cmap=cmap,
            interpolation="none",
            vmax=wide_vmax,
            extent=[-1, 1, -1, 1],
        )
        axes[1].imshow(
            zoom_map[::-1],
            cmap=cmap,
            interpolation="none",
            vmax=zoom_vmax,
            extent=[-1, 1, -1, 1],
        )

        for ax in axes:
            ax.scatter(0, 0, color="gray", alpha=0.8, s=256)
            ax.scatter(0, 0, color="r", marker="x", linewidth=2, s=128)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("gray")
            ax.set_rasterized(True)
            cbar = fig.colorbar(
                ax.get_images()[0],
                ax=ax,
                location="bottom",
                shrink=0.8,
            )
            cbar.set_label("height [m]")

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
