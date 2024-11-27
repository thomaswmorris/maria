from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import scipy as sp

from ..map import ProjectedMap
from ..tod import TOD

# np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)


class BaseMapper:
    """
    The base class for mapping.
    """

    def __init__(
        self,
        center: tuple[float, float] = (0, 0),
        width: float = 1,
        height: float = 1,
        resolution: float = 0.01,
        frame: str = "ra_dec",
        units: str = "K_RJ",
        degrees: bool = True,
        calibrate: bool = False,
        tods: Sequence[TOD] = [],
        verbose: bool = True,
    ):
        self.resolution = np.radians(resolution) if degrees else resolution
        self.center = np.radians(center) if degrees else center
        self.width = np.radians(width) if degrees else width
        self.height = np.radians(height) if degrees else height
        self.degrees = degrees
        self.calibrate = calibrate
        self.frame = frame
        self.units = units

        self.verbose = verbose

        self.n_x = int(np.maximum(1, self.width / self.resolution))
        self.n_y = int(np.maximum(1, self.height / self.resolution))

        self.x_bins = np.linspace(-0.5 * self.width, 0.5 * self.width, self.n_x + 1)
        self.y_bins = np.linspace(-0.5 * self.height, 0.5 * self.height, self.n_y + 1)

        self.tods = []
        self.add_tods(tods)

    def plot(self):
        if not hasattr(self, "map"):
            raise RuntimeError("Mapper has not been run yet.")
        self.map.plot()

    @property
    def n_maps(self):
        return len(self.maps)

    def add_tods(self, tods):
        for tod in np.atleast_1d(tods):
            self.tods.append(tod)
        self.bands = list(
            np.unique([list(np.unique(tod.dets.band_name)) for tod in self.tods]),
        )

    def _run(self):
        raise ValueError("Not implemented!")

    def run(self):
        self.map_data = {}

        for band in self.bands:
            self.map_data[band] = self._run(band)

        map_data = np.zeros((len(self.map_data), 1, self.n_y, self.n_x))
        map_weight = np.zeros((len(self.map_data), 1, self.n_y, self.n_x))
        map_freqs = []

        for i, (band_name, band_map_data) in enumerate(self.map_data.items()):
            map_freqs.append(band_map_data["nom_freq"])

            band_map_numer = band_map_data["sum"].copy()
            band_map_denom = band_map_data["weight"].copy()

            if "gaussian_filter" in self.map_postprocessing.keys():
                sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

                band_map_numer = sp.ndimage.gaussian_filter(band_map_numer, sigma=sigma)
                band_map_denom = sp.ndimage.gaussian_filter(band_map_denom, sigma=sigma)

            if "median_filter" in self.map_postprocessing.keys():
                band_map_numer = sp.ndimage.median_filter(
                    band_map_numer,
                    size=self.map_postprocessing["median_filter"]["size"],
                )

            map_data[i, :] = band_map_numer / band_map_denom
            map_weight[i, :] = band_map_denom

        return ProjectedMap(
            data=map_data,
            weight=map_weight,
            nu=map_freqs,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
        )
