from __future__ import annotations

import logging
import os
import time as ttime
import warnings
from collections.abc import Sequence
from typing import Mapping

import numpy as np
import scipy as sp

from ..coords import FRAMES, Frame, get_center_phi_theta
from ..instrument import BandList
from ..io import humanize_time
from ..map import ProjectedMap
from ..tod import TOD
from ..units import Quantity

# np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BaseProjectionMapper:
    """
    The base class for mapping.
    """

    def __init__(
        self,
        center: tuple[float, float],
        stokes: str,
        width: float,
        height: float,
        resolution: float,
        frame: str,
        units: str,
        degrees: bool,
        calibrate: bool,
        tods: Sequence[TOD],
        tod_preprocessing: Mapping,
        map_postprocessing: Mapping,
    ):
        center = center or np.degrees(get_center_phi_theta(*np.stack([tod.coords.center(frame="ra/dec") for tod in tods]).T))
        height = height or width

        self.resolution = Quantity(resolution, "deg" if degrees else "rad")
        self.center = Quantity(center, "deg" if degrees else "rad")
        self.width = Quantity(width, "deg" if degrees else "rad")
        self.height = Quantity(height, "deg" if degrees else "rad")
        self.degrees = degrees
        self.calibrate = calibrate
        self.frame = Frame(frame)
        self.units = units
        self.stokes = stokes
        self.tod_preprocessing = tod_preprocessing
        self.map_postprocessing = map_postprocessing

        self.bands = BandList([])

        self.tods = []
        self.add_tods(tods)

        self.map = ProjectedMap(
            data=np.zeros((self.n_y, self.n_x)),
            weight=np.ones((self.n_y, self.n_x)),
            stokes=self.stokes,
            nu=self.bands.center,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.units,
        )

    def add_tods(self, tods):
        for tod in np.atleast_1d(tods):
            self.tods.append(tod.process(config=self.tod_preprocessing).to(self.units))

            for band in tod.dets.bands:
                self.bands.add(band)

        self.initialize_mapper()

    def initialize_mapper(self):
        raise NotImplementedError()

    @property
    def map_shape(self):
        return (len(self.stokes), self.n_x, self.n_y)

    @property
    def n_x(self):
        return int(np.maximum(1, self.width / self.resolution))

    @property
    def n_y(self):
        return int(np.maximum(1, self.height / self.resolution))

    @property
    def x_bins(self):
        return np.linspace(-0.5 * self.width.rad, 0.5 * self.width.rad, self.n_x + 1)

    @property
    def y_bins(self):
        return np.linspace(0.5 * self.height.rad, -0.5 * self.height.rad, self.n_y + 1)

    def plot(self):
        if not hasattr(self, "map"):
            raise RuntimeError("Mapper has not been run yet.")
        self.map.plot()

    @property
    def n_stokes(self):
        return len(self.stokes)

    @property
    def n_bands(self):
        return len(self.bands)

    def _run(self):
        raise ValueError("Not implemented!")

    def run(self):
        if not len(self.tods):
            raise RuntimeError("This mapper has no TODs!")

        map_data = self._run()

        if "gaussian_filter" in self.map_postprocessing.keys():
            sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

            map_data["sum"] = sp.ndimage.gaussian_filter(map_data["sum"], sigma=(0, 0, sigma, sigma))
            map_data["wgt"] = sp.ndimage.gaussian_filter(map_data["wgt"], sigma=(0, 0, sigma, sigma))

        if "median_filter" in self.map_postprocessing.keys():
            size = self.map_postprocessing["median_filter"]["size"]
            map_data["sum"] = sp.ndimage.median_filter(
                map_data["sum"],
                size=size,
                axes=(-2, -1),
            )
            map_data["wgt"] = sp.ndimage.median_filter(
                map_data["wgt"],
                size=size,
                axes=(-2, -1),
            )

        for stokes_index, stokes in enumerate(self.stokes):
            for nu_index, nu in enumerate(self.bands.center):
                if map_data["wgt"][stokes_index, nu_index].sum() == 0:
                    logger.warning(f"No counts for map (stokes={stokes}, nu={Quantity(nu, 'Hz')})")

        data = map_data["sum"] / map_data["wgt"]

        # by convention maps have zero mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data_offsets = np.nanmean(data, axis=(-1, -2))[..., None, None]

        return ProjectedMap(
            data=(data - data_offsets),
            stokes=self.stokes,
            weight=map_data["wgt"],
            nu=self.bands.center,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.units,
        )
