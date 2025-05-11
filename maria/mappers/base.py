from __future__ import annotations

import logging
import os
import time as ttime
import warnings
from collections.abc import Sequence

import numpy as np
import scipy as sp

from ..instrument import BandList
from ..io import humanize_time
from ..map import ProjectedMap
from ..tod import TOD
from ..units import Quantity

# np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BaseMapper:
    """
    The base class for mapping.
    """

    def __init__(
        self,
        center: tuple[float, float],
        stokes: float,
        width: float,
        height: float,
        resolution: float,
        frame: str,
        units: str,
        degrees: bool,
        calibrate: bool,
        tods: Sequence[TOD],
    ):
        self.resolution = np.radians(resolution) if degrees else resolution
        self.center = np.radians(center) if degrees else center
        self.width = np.radians(width) if degrees else width
        self.height = np.radians(height) if degrees else height
        self.degrees = degrees
        self.calibrate = calibrate
        self.frame = frame
        self.units = units
        self.stokes = stokes

        self.n_x = int(np.maximum(1, self.width / self.resolution))
        self.n_y = int(np.maximum(1, self.height / self.resolution))

        self.x_bins = np.linspace(-0.5 * self.width, 0.5 * self.width, self.n_x + 1)
        self.y_bins = np.linspace(0.5 * self.height, -0.5 * self.height, self.n_y + 1)

        self.bands = BandList(bands=[])

        self.tods = []
        self.add_tods(tods)

    def plot(self):
        if not hasattr(self, "map"):
            raise RuntimeError("Mapper has not been run yet.")
        self.map.plot()

    @property
    def n_stokes(self):
        return len(self.stokes)

    def add_tods(self, tods):
        for tod in np.atleast_1d(tods):
            self.tods.append(tod)

            for band in tod.dets.bands:
                self.bands.add(band)

        # self.bands = list(
        #     np.unique([list(np.unique(tod.dets.band_name)) for tod in self.tods]),
        # )

    def _run(self):
        raise ValueError("Not implemented!")

    def run(self):
        if not len(self.tods):
            raise RuntimeError("This mapper has no TODs!")

        self.map_data = {}

        for band in self.bands:
            band_start_s = ttime.monotonic()
            self.map_data[band.name] = self._run(band)
            logger.info(f"Ran mapper for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

        map_data = np.zeros((self.n_stokes, len(self.bands), 1, self.n_y, self.n_x))
        map_weight = np.zeros((self.n_stokes, len(self.bands), 1, self.n_y, self.n_x))
        map_freqs = []

        for i, (band_name, band_map_data) in enumerate(self.map_data.items()):
            map_freqs.append(band_map_data["nom_freq"])

            band_map_numer = band_map_data["sum"].copy()
            band_map_denom = band_map_data["weight"].copy()

            if "gaussian_filter" in self.map_postprocessing.keys():
                sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

                band_map_numer = sp.ndimage.gaussian_filter(band_map_numer, sigma=(0, 0, sigma, sigma))
                band_map_denom = sp.ndimage.gaussian_filter(band_map_denom, sigma=(0, 0, sigma, sigma))

            if "median_filter" in self.map_postprocessing.keys():
                size = self.map_postprocessing["median_filter"]["size"]
                band_map_numer = sp.ndimage.median_filter(
                    band_map_numer,
                    size=size,
                    axes=(-2, -1),
                )

            map_data[:, i] = band_map_numer / band_map_denom
            map_weight[:, i] = band_map_denom

        for stokes_index, stokes in enumerate(self.stokes):
            for nu_index, nu in enumerate(map_freqs):
                if map_weight[stokes_index, nu_index].sum() == 0:
                    logger.warning(f"No counts for map (stokes={stokes}, nu={Quantity(nu, 'Hz')})")

        # by convention maps have zero mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            map_offsets = np.nanmean(map_data, axis=(-1, -2))[..., None, None]

        return ProjectedMap(
            data=map_data - map_offsets,
            stokes=self.stokes,
            weight=map_weight,
            nu=map_freqs,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame,
            units=self.units,
        )
