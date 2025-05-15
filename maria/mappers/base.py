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

        map_numer = np.zeros((self.n_stokes, len(self.bands), self.n_y, self.n_x))
        map_denom = np.zeros((self.n_stokes, len(self.bands), self.n_y, self.n_x))

        for band_index, band in enumerate(self.bands):
            band_start_s = ttime.monotonic()
            band_maps = self._run(band)

            map_numer[:, band_index] = band_maps["numer"]
            map_denom[:, band_index] = band_maps["denom"]

            logger.info(f"Ran mapper for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

        if "gaussian_filter" in self.map_postprocessing.keys():
            sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

            map_numer = sp.ndimage.gaussian_filter(map_numer, sigma=(0, 0, sigma, sigma))
            map_denom = sp.ndimage.gaussian_filter(map_denom, sigma=(0, 0, sigma, sigma))

        if "median_filter" in self.map_postprocessing.keys():
            size = self.map_postprocessing["median_filter"]["size"]
            map_numer = sp.ndimage.median_filter(
                map_numer,
                size=size,
                axes=(-2, -1),
            )
            map_denom = sp.ndimage.median_filter(
                map_denom,
                size=size,
                axes=(-2, -1),
            )

        for stokes_index, stokes in enumerate(self.stokes):
            for nu_index, nu in enumerate(self.bands.center):
                if map_denom[stokes_index, nu_index].sum() == 0:
                    logger.warning(f"No counts for map (stokes={stokes}, nu={Quantity(nu, 'Hz')})")

        map_data = map_numer / map_denom

        # by convention maps have zero mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            map_offsets = np.nanmean(map_data, axis=(-1, -2))[..., None, None]

        return ProjectedMap(
            data=(map_data - map_offsets),
            stokes=self.stokes,
            weight=map_denom,
            nu=self.bands.center,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame,
            units=self.units,
        )
