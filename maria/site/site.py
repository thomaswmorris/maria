from __future__ import annotations

import os
from pathlib import Path

import h5py
import healpy as hp
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation
from matplotlib import pyplot as plt

from ..io import fetch, repr_lat_lon
from ..units import Quantity

here, this_filename = os.path.split(__file__)

REGION_DISPLAY_COLUMNS = ["location", "country", "latitude", "longitude", "timezone"]
REGIONS = pd.read_csv(f"{here}/regions.csv", index_col=0)
all_regions = list(REGIONS.index.values)


class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        super().__init__(
            f"The region '{invalid_region}' is not supported. "
            f"Supported regions are:\n\n{REGIONS.loc[:, REGION_DISPLAY_COLUMNS].to_string()}",
        )


def get_height_map():
    with h5py.File(fetch("world_heightmap.h5"), "r") as f:
        height_map = f["data"][:].astype(np.uint16)
    return 32 * np.where(height_map < 255, height_map, np.nan)


class Site:
    """
    An object representing a point on the earth that parametrizes observing conditions.
    """

    def __init__(
        self,
        description: str = "",
        region: str = "princeton",
        timezone: str = "UTC",
        seasonal: bool = True,
        diurnal: bool = True,
        latitude: float = None,  # in degrees
        longitude: float = None,  # in degrees
        altitude: float = None,  # in meters
        weather_quantiles: dict = {},
        documentation: str = "",
        instruments: list = [],
    ):
        if region not in REGIONS.index.values:
            raise InvalidRegionError(region)

        if longitude is None:
            longitude = Quantity(REGIONS.loc[region].longitude, "deg")

        if latitude is None:
            latitude = Quantity(REGIONS.loc[region].latitude, "deg")

        if altitude is None:
            altitude = Quantity(REGIONS.loc[region].altitude, "m")

        self.description = description
        self.region = region
        self.timezone = REGIONS.loc[region, "timezone"] if region in all_regions else timezone
        self.seasonal = seasonal
        self.diurnal = diurnal
        self.latitude = Quantity(latitude, "deg")
        self.longitude = Quantity(longitude, "deg")
        self.altitude = Quantity(altitude, "m")
        self.weather_quantiles = weather_quantiles
        self.documentation = documentation
        self.instruments = instruments

        self.earth_location = EarthLocation.from_geodetic(
            lon=self.longitude.deg,
            lat=self.latitude.deg,
            height=self.altitude.m,
        )

    def plot(self, res=0.03, wide_size: float = 45, zoom_size: float = 2):
        height_map = get_height_map()

        kwargs = {
            "rot": (self.longitude.deg, self.latitude.deg),
            "cmap": "gist_earth",
            "flip": "geo",
            "badcolor": "gray",
            "reso": 60 * res,
            "return_projected_map": True,
            "no_plot": True,
        }

        zoom_map = hp.gnomview(
            height_map,
            xsize=zoom_size / res,
            **kwargs,
        )  # , norm=mpl.colors.Normalize(vmin=0, vmax=5e3))
        wide_map = hp.gnomview(
            height_map,
            xsize=wide_size / res,
            **kwargs,
        )  # , norm=mpl.colors.Normalize(vmin=0, vmax=5e3))

        fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)

        zoom_vmax = np.nanpercentile(zoom_map.data, q=99)
        wide_vmax = np.nanpercentile(wide_map.data, q=99)

        cmap = mpl.cm.gist_ncar
        cmap = mpl.cm.gist_earth
        cmap.set_bad("royalblue", 1.0)

        wide_mesh = axes[0].pcolormesh(
            wide_size * np.linspace(-0.5, 0.5, int(wide_size / res)),
            wide_size * np.linspace(-0.5, 0.5, int(wide_size / res)),
            wide_map,
            cmap=cmap,
            vmax=wide_vmax,
        )
        cbar = fig.colorbar(wide_mesh, ax=axes[0], location="bottom", shrink=0.8, label="height")
        cbar.set_label("height [m]")

        zoom_mesh = axes[1].pcolormesh(
            zoom_size * np.linspace(-0.5, 0.5, int(zoom_size / res)),
            zoom_size * np.linspace(-0.5, 0.5, int(zoom_size / res)),
            zoom_map,
            cmap=cmap,
            vmax=zoom_vmax,
        )
        cbar = fig.colorbar(zoom_mesh, ax=axes[1], location="bottom", shrink=0.8, label="height")
        cbar.set_label("height [m]")

        for ax in axes:
            # ax.scatter(0, 0, color="gray", alpha=0.8, s=256)
            ax.scatter(0, 0, edgecolor="r", facecolor="none", linewidth=2, s=256)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("gray")
            ax.set_rasterized(True)

    @property
    def location(self):
        return (self.longitude, self.latitude, self.altitude)

    def __repr__(self):
        repr_lat, repr_lon = repr_lat_lon(self.latitude.degrees, self.longitude.degrees)
        s = f"""Site:
  region: {self.region}
  timezone: {self.timezone}
  location:
    longitude: {repr_lon}
    latitude:  {repr_lat}
    altitude: {self.altitude}
  seasonal: {self.seasonal}
  diurnal: {self.diurnal}"""

        return s
