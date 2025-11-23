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

from ..coords import FRAMES, Frame, infer_center_width_height
from ..instrument import BandList
from ..io import DEFAULT_BAR_FORMAT, repr_phi_theta
from ..map import MAP_QUANTITIES, ProjectionMap
from ..tod import TOD, TOD_QUANTITIES
from ..units import Quantity, parse_units

# np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BaseMapper:
    """
    The base class for mapping, which handles both projection maps and HEALPix maps.
    """

    def __init__(
        self,
        tods: Sequence[TOD],
        resolution: Quantity,
        units: str,
        stokes: str,
        min_time: float,
        max_time: float,
        timestep: float,
        tod_preprocessing: Mapping,
        map_postprocessing: Mapping,
        progress_bars: bool,
    ):
        u = parse_units(units)
        self.tod_units = units if u["quantity"] in TOD_QUANTITIES else "K_RJ"
        self.map_units = "K_RJ"

        if stokes is None:
            stokes = "IQUV" if any([tod.dets.polarized for tod in tods]) else "I"
            logger.info(f"Inferring mapper stokes parameters '{stokes}' for mapper.")

        if u["quantity"] not in MAP_QUANTITIES:
            raise ValueError(f"Units '{units}' (with associated quantity '{u['quantity']}') are not valid map units")

        self.resolution = resolution
        self.units = units
        self.stokes = stokes
        self.tod_preprocessing = tod_preprocessing
        self.map_postprocessing = map_postprocessing
        self.progress_bars = progress_bars

        min_time = min_time or min([tod.coords.t.min() for tod in tods])
        max_time = max_time or max([tod.coords.t.max() for tod in tods])
        mean_time = (min_time + max_time) / 2

        if timestep is None:
            self.timestep = np.inf
            self.t = np.array([mean_time])
        else:
            self.timestep = timestep
            time_bins = np.arange(min_time, max_time + timestep, timestep)
            time_bins += mean_time - self.time_bins.mean()
            self.t = (time_bins[1:] + time_bins[:-1]) / 2

        self.bands = BandList([])

        self.tods = []
        self.add_tods(tods)

        beam_sum = np.zeros((len(self.bands), 1, 3))
        beam_wgt = np.zeros((len(self.bands), 1, 3))

        for band_index, band in enumerate(self.bands):
            for tod in self.tods:
                beam_sum[band_index] += tod.duration * tod.dets.beams[tod.dets.band_name == band.name].mean(axis=0)
                beam_wgt[band_index] += tod.duration

        self.beam = beam_sum / beam_wgt

    @property
    def nu(self):
        return np.unique([nu.Hz for nu in self.bands.center])

    @property
    def nu_bins(self):
        return np.array([0, *(self.nu[1:] + self.nu[:-1]), np.inf])

    @property
    def t_bins(self):
        return np.array([-np.inf, *(self.t[1:] + self.t[:-1]), np.inf])

    @property
    def n_stokes(self):
        return len(self.stokes)

    @property
    def n_bands(self):
        return len(self.bands)

    @property
    def n_t(self):
        return len(self.t)

    @property
    def map(self):
        return ProjectionMap(
            data=self.data.reshape(self.map_shape),
            weight=self.weight.reshape(self.map_shape),
            stokes=self.stokes,
            t=self.timestamps,
            nu=self.bands.center,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.tod_units,
            beam=self.beam,
        ).to(self.map_units)

    def add_tods(self, tods):
        tods_pbar = tqdm(
            np.atleast_1d(tods), desc="Preprocessing TODs", bar_format=DEFAULT_BAR_FORMAT, disable=not self.progress_bars
        )
        for tod in tods_pbar:
            self.tods.append(tod.process(config=self.tod_preprocessing).to(self.tod_units))
            for band in tod.dets.bands:
                self.bands.add(band)

    def initialize_mapper(self):
        raise NotImplementedError()

    def plot(self):
        if not hasattr(self, "map"):
            raise RuntimeError("Mapper has not been run yet.")
        self.map.plot()

    def _run(self):
        """
        A method to be overwritten by the specific mapping procedure.

        It should return a dict with the raw map and weight.
        """
        raise ValueError("Not implemented!")

    def run(self):
        if not len(self.tods):
            raise RuntimeError("This mapper has no TODs!")

        map_data = self._run()
        map_data["sum"] = map_data["data"] * map_data["weight"]

        # if "median_filter" in self.map_postprocessing.keys():
        #     size = self.map_postprocessing["median_filter"]["size"]
        #     map_data["data"] = sp.ndimage.median_filter(
        #         map_data["data"],
        #         size=size,
        #         axes=(-2, -1),
        #     )

        if "gaussian_filter" in self.map_postprocessing.keys():
            sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

            map_data["sum"] = sp.ndimage.gaussian_filter(map_data["sum"], sigma=(0, 0, 0, sigma, sigma))
            map_data["weight"] = sp.ndimage.gaussian_filter(map_data["weight"], sigma=(0, 0, 0, sigma, sigma))

        self.data = (map_data["sum"] / map_data["weight"]).reshape(self.map_shape)
        self.weight = map_data["weight"].reshape(self.map_shape)

        for stokes_index, stokes in enumerate(self.stokes):
            for nu_index, nu in enumerate(self.nu):
                if map_data["weight"][stokes_index, nu_index].sum() == 0:
                    logger.warning(f"No counts for map (stokes={stokes}, nu={Quantity(nu, 'Hz')})")

        # by convention maps have zero mean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.data -= np.nanmean(self.data, axis=(-1, -2))[..., None, None]

        return self.map

    @property
    def map(self):
        raise NotImplementedError()


class BaseProjectionMapper(BaseMapper):
    """
    The base class for making a Projection map.
    """

    def __init__(
        self,
        tods: Sequence[TOD],
        center: tuple[Quantity, Quantity],
        stokes: str,
        width: Quantity,
        height: Quantity,
        resolution: Quantity,
        frame: str,
        units: str,
        tod_preprocessing: Mapping,
        map_postprocessing: Mapping,
        min_time: float,
        max_time: float,
        timestep: float,
        degrees: bool,
        progress_bars: bool,
    ):
        center = (Quantity(center, "deg" if degrees else "rad")) if center is not None else None
        width = (Quantity(width, "deg" if degrees else "rad")) if width is not None else None
        height = (Quantity(height, "deg" if degrees else "rad")) if height is not None else None
        resolution = (Quantity(resolution, "deg" if degrees else "rad")) if resolution is not None else None

        infer_center, infer_width, infer_height = infer_center_width_height(
            coords_list=[tod.coords for tod in tods], center=center, frame=frame, square=True
        )

        logger.debug(
            f"Inferred center={Quantity(infer_center, 'rad')}, width={Quantity(infer_width, 'rad')}, \
                     width={Quantity(infer_height, 'rad')} for map."
        )

        if center is None:
            center = Quantity(infer_center, "rad")
            logger.info(
                f"Inferring center {repr_phi_theta(phi=center[0].rad, theta=center[1].rad, frame=frame)} for mapper."
            )

        if width is None:
            if height is not None:
                width = height
                logger.info(f"Inferring mapper width {width} to match supplied height.")
            else:
                width = Quantity(infer_width, "rad")
                logger.info(f"Inferring mapper width {width} for mapper from observation patch.")

        if height is None:
            if width is not None:
                height = width
                logger.info(f"Inferring mapper height {height} to match supplied width.")
            else:
                height = Quantity(infer_height, "rad")
                logger.info(f"Inferring mapper height {height} for mapper from observation patch.")

        if resolution is None:
            resolution = Quantity(width / 128, "rad")
            logger.info(f"Inferring mapper resolution {resolution} for mapper from observation patch.")

        super().__init__(
            tods=tods,
            resolution=resolution,
            units=units,
            stokes=stokes,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
            progress_bars=progress_bars,
        )

        self.center = center
        self.width = width
        self.height = height
        self.frame = Frame(frame)

        self.data = np.nan * np.zeros(self.map_shape)
        self.weight = np.nan * np.ones(self.map_shape)

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

    @property
    def map_shape(self):
        return (self.n_stokes, len(self.nu), self.n_t, self.n_y, self.n_x)

    @property
    def map_size(self):
        return np.prod(self.map_shape)

    @property
    def map(self):
        return ProjectionMap(
            data=self.data,
            weight=self.weight,
            stokes=self.stokes,
            t=self.t,
            nu=self.nu,
            resolution=self.resolution,
            center=self.center,
            degrees=False,
            frame=self.frame.name,
            units=self.tod_units,
            beam=self.beam,
        ).to(self.map_units)
