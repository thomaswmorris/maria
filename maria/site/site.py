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

from ..constants import EARTH_RADIUS
from ..io import fetch, repr_lat_lon
from ..units import Quantity
from .height_map import get_height_map
from .region import REGIONS, InvalidRegionError, all_regions  # noqa

here, this_filename = os.path.split(__file__)


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

    def plot(self, res=0.03, wide_size: float = 25, zoom_size: float = 2, cmap="cubehelix"):
        height_map = get_height_map()

        kwargs = {
            "rot": (self.longitude.deg, self.latitude.deg),
            "cmap": cmap,
            "flip": "geo",
            "badcolor": "gray",
            "reso": 60 * res,
            "return_projected_map": True,
            "no_plot": True,
        }

        wide_map = hp.gnomview(
            height_map,
            xsize=wide_size / res,
            **kwargs,
        )[::2, ::2]

        zoom_map = hp.gnomview(
            height_map,
            xsize=zoom_size / res,
            **kwargs,
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)

        wide_vmax = np.nanpercentile(wide_map.data, q=99.9)
        zoom_vmax = np.nanpercentile(zoom_map.data, q=95)

        cmap = plt.get_cmap(cmap)
        cmap.set_bad("royalblue", 1.0)

        wide_mesh = axes[0].pcolormesh(
            1e-3 * EARTH_RADIUS * np.radians(wide_size) * np.linspace(-0.5, 0.5, wide_map.shape[0]),
            1e-3 * EARTH_RADIUS * np.radians(wide_size) * np.linspace(-0.5, 0.5, wide_map.shape[1]),
            wide_map,
            cmap=cmap,
            vmax=wide_vmax,
            shading="nearest",
        )
        cbar = fig.colorbar(wide_mesh, ax=axes[0], location="bottom", shrink=0.9, label="height")
        cbar.set_label("Height [meters]")

        zoom_mesh = axes[1].pcolormesh(
            1e-3 * EARTH_RADIUS * np.radians(zoom_size) * np.linspace(-0.5, 0.5, zoom_map.shape[0]),
            1e-3 * EARTH_RADIUS * np.radians(zoom_size) * np.linspace(-0.5, 0.5, zoom_map.shape[1]),
            zoom_map,
            cmap=cmap,
            vmax=zoom_vmax,
            shading="nearest",
        )
        cbar = fig.colorbar(zoom_mesh, ax=axes[1], location="bottom", shrink=0.9, label="height")
        cbar.set_label("Height [meters]")

        for ax in axes:
            ax.scatter(0, 0, edgecolor="r", facecolor="none", linewidth=2, s=256)
            ax.set_xlabel("Distance [kilometers]")
            ax.set_yticks([])
            ax.set_facecolor("gray")
            ax.set_rasterized(True)
            ax.set_aspect("equal")

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
