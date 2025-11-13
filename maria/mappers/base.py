from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from typing import Mapping

import arrow
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..coords import FRAMES, Frame, get_center_phi_theta, infer_center_width_height
from ..instrument import BandList
from ..io import DEFAULT_BAR_FORMAT
from ..map import MAP_QUANTITIES, ProjectedMap
from ..tod import TOD, TOD_QUANTITIES
from ..units import Quantity, parse_units

# np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BaseProjectionMapper:
    """
    The base class for mapping.
    """

    def __init__(
        self,
        center: tuple[Quantity, Quantity],
        stokes: str,
        width: Quantity,
        height: Quantity,
        resolution: Quantity,
        frame: str,
        units: str,
        calibrate: bool,
        tods: Sequence[TOD],
        tod_preprocessing: Mapping,
        map_postprocessing: Mapping,
        min_time: float,
        max_time: float,
        timestep: float,
    ):
        u = parse_units(units)
        self.tod_units = units if u["quantity"] in TOD_QUANTITIES else "K_RJ"
        self.map_units = "K_RJ"

        if u["quantity"] not in MAP_QUANTITIES:
            raise ValueError(f"Units '{units}' (with associated quantity '{u['quantity']}') are not valid map units")

        self.resolution = resolution
        self.center = center
        self.width = width
        self.height = height
        self.calibrate = calibrate
        self.frame = Frame(frame)
        self.units = units
        self.stokes = stokes
        self.tod_preprocessing = tod_preprocessing
        self.map_postprocessing = map_postprocessing

        if timestep is None:
            self.timestamps = np.mean([min_time, max_time])
        else:
            self.time_bins = np.arange(min_time, max_time + timestep, timestep)
            self.timestamps = (self.time_bins[1:] + self.time_bins[:-1]) / 2

        self.bands = BandList([])

        self.tods = []
        self.add_tods(tods)

        self.map = ProjectedMap(
            data=np.zeros((self.n_y, self.n_x)),
            weight=np.ones((self.n_y, self.n_x)),
            stokes=self.stokes,
            t=self.timestamps,
            nu=self.bands.center,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.units,
        )

    def add_tods(self, tods):
        tods_pbar = tqdm(np.atleast_1d(tods), desc="Preprocessing TODs", bar_format=DEFAULT_BAR_FORMAT)
        for tod in tods_pbar:
            self.tods.append(tod.process(config=self.tod_preprocessing).to(self.tod_units))
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

            map_data["sum"] = sp.ndimage.gaussian_filter(map_data["sum"], sigma=(0, 0, 0, sigma, sigma))
            map_data["wgt"] = sp.ndimage.gaussian_filter(map_data["wgt"], sigma=(0, 0, 0, sigma, sigma))

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

        beam_sum = np.zeros((len(self.bands), 1, 3))
        beam_wgt = np.zeros((len(self.bands), 1, 3))

        for band_index, band in enumerate(self.bands):
            for tod in self.tods:
                beam_sum[band_index] += tod.duration * tod.dets.beams[tod.dets.band_name == band.name].mean(axis=0)
                beam_wgt[band_index] += tod.duration

        return ProjectedMap(
            data=(data - data_offsets),
            stokes=self.stokes,
            weight=map_data["wgt"],
            nu=self.bands.center,
            t=self.timestamps,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.tod_units,
            beam=beam_sum / beam_wgt,
        ).to(self.units)
