from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import numpy as np
from tqdm import tqdm

from ..atmosphere import DEFAULT_ATMOSPHERE_KWARGS, Atmosphere
from ..cmb import CMB, DEFAULT_CMB_KWARGS, generate_cmb, get_cmb
from ..errors import PointingError
from ..instrument import Instrument
from ..io import fetch, humanize_time
from ..map import Map, load
from ..plan import Plan
from ..site import Site
from .atmosphere import AtmosphereMixin
from .base import BaseSimulation
from .cmb import CMBMixin
from .map import MapMixin
from .noise import NoiseMixin

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

MIN_ELEVATION_WARN = 10  # degrees
MIN_ELEVATION_ERROR = 5  # degrees


class Simulation(BaseSimulation, AtmosphereMixin, CMBMixin, MapMixin, NoiseMixin):
    """
    A simulation of a telescope. This is what users should touch, primarily.
    """

    @classmethod
    def from_config(cls, config: dict = {}, **params):
        return cls(**params)

    def __init__(
        self,
        instrument: tuple[Instrument, str] = "test/1deg",
        plan: tuple[Plan, str] = "one_minute_zenith_stare",
        site: tuple[Site, str] = "hoagie_haven",
        atmosphere: tuple[Atmosphere, str] = None,
        cmb: tuple[CMB, str] = None,
        map: tuple[Map, str] = None,
        noise: bool = True,
        atmosphere_kwargs: dict = {},
        cmb_kwargs: dict = {},
        map_kwargs: dict = {},
        noise_kwargs: dict = {},
        progress_bars: bool = True,
        keep_mean_signal: bool = False,
    ):
        super().__init__(
            instrument=instrument,
            plan=plan,
            site=site,
            progress_bars=progress_bars,
            keep_mean_signal=keep_mean_signal,
        )

        sim_start_s = ttime.monotonic()

        self.noise = noise

        self.map_kwargs = map_kwargs
        self.noise_kwargs = noise_kwargs

        self.start = arrow.get(self.boresight.t.min()).to("utc")
        self.end = arrow.get(self.boresight.t.max()).to("utc")

        if atmosphere:
            self.atmosphere_kwargs = DEFAULT_ATMOSPHERE_KWARGS.copy()
            self.atmosphere_kwargs.update(atmosphere_kwargs)

            # do some checks
            el_min = np.atleast_1d(self.coords.el).min()
            if el_min < np.radians(MIN_ELEVATION_WARN):
                logger.warning(
                    f"Some ArrayList come within {MIN_ELEVATION_WARN} degrees of the horizon"
                    f"(el_min = {np.degrees(el_min):.01f}°)",
                )
            if el_min <= np.radians(MIN_ELEVATION_ERROR):
                raise PointingError(
                    f"Some ArrayList come within {MIN_ELEVATION_ERROR} degrees of the horizon"
                    f"(el_min = {np.degrees(el_min):.01f}°)",
                )

            self.weather_kwargs = self.atmosphere_kwargs.pop("weather") if "weather" in self.atmosphere_kwargs else {}

            self.atmosphere = Atmosphere(
                model=atmosphere,
                timestamp=self.plan.time.mean(),
                region=self.site.region,
                altitude=self.site.altitude,
                weather_kwargs=self.weather_kwargs,
                disable_progress_bars=self.disable_progress_bars,
                **self.atmosphere_kwargs,
            )

            # give it the simulation, so that it knows about pointing, site, etc. (kind of cursed)
            self.atmosphere.initialize(self)

            logger.debug(f"Initialized atmosphere simulation in {humanize_time(ttime.monotonic() - sim_start_s)}.")

        cmb_start_s = ttime.monotonic()

        if cmb:
            self.cmb_kwargs = DEFAULT_CMB_KWARGS.copy()
            self.cmb_kwargs.update(cmb_kwargs)

            if cmb in ["spectrum", "power_spectrum", "generate", "generated"]:
                for _ in tqdm(
                    range(1),
                    desc=f"Generating CMB (nside={self.cmb_kwargs['nside']})",
                    disable=self.disable_progress_bars,
                ):
                    self.cmb = generate_cmb(**self.cmb_kwargs)
            elif cmb in ["real", "planck"]:
                self.cmb = get_cmb(**self.cmb_kwargs)
            else:
                raise ValueError(f"Invalid value for cmb: '{cmb}'.")

            logger.debug(f"Initialized CMB simulation in {humanize_time(ttime.monotonic() - cmb_start_s)}.")

        map_start_s = ttime.monotonic()
        if map:
            if isinstance(map, str):
                map = load(fetch(map), **map_kwargs)
            elif not isinstance(map, Map):
                raise ValueError("")

            # the map can be frequency-naive if it is already in K_RJ
            self.map = map.to(units="K_RJ")

            if "stokes" not in self.map.dims:
                self.map = self.map.unsqueeze("stokes")

            if "nu" not in self.map.dims:
                self.map = self.map.unsqueeze("nu")

            if "t" in self.map.dims:
                if self.map.dims["t"] > 1:
                    map_start = arrow.get(self.map.t.min()).to("utc")
                    map_end = arrow.get(self.map.t.max()).to("utc")
                    if map_start > self.start:
                        logger.warning(
                            f"Beginning of map ({map_start.isoformat()[:26]}) is after the "
                            f"beginning of the simulation ({self.start.isoformat()[:26]}).",
                        )
                    if map_end < self.end:
                        logger.warning(
                            f"End of map ({map_end.isoformat()[:26]}) is before the "
                            f"end of the simulation ({self.end.isoformat()[:26]}).",
                        )
                else:
                    self.map = self.map.squeeze("t")

        logger.debug(f"Initialized map simulation in {humanize_time(ttime.monotonic() - map_start_s)}.")

        noise_start_s = ttime.monotonic()
        if noise:
            pass
        logger.debug(f"Initialized noise simulation in {humanize_time(ttime.monotonic() - noise_start_s)}.")

        logger.debug(f"Initialized simulation in {humanize_time(ttime.monotonic() - sim_start_s)}.")

    def _run(self):
        if hasattr(self, "atmosphere"):
            atmosphere_sim_start_s = ttime.monotonic()
            self._simulate_atmosphere()
            self._compute_atmospheric_emission()
            logger.debug(f"Ran atmosphere simulation in {humanize_time(ttime.monotonic() - atmosphere_sim_start_s)}.")

        if hasattr(self, "cmb"):
            cmb_sim_start_s = ttime.monotonic()
            self._simulate_cmb_emission()
            logger.debug(f"Ran CMB simulation in {humanize_time(ttime.monotonic() - cmb_sim_start_s)}.")

        if hasattr(self, "map"):
            map_sim_start_s = ttime.monotonic()
            self._sample_maps()
            logger.debug(f"Ran map simulation in {humanize_time(ttime.monotonic() - map_sim_start_s)}.")

        # number of bands are lost here
        if self.noise:
            noise_sim_start_s = ttime.monotonic()
            self._simulate_noise()
            logger.debug(f"Ran noise simulation in {humanize_time(ttime.monotonic() - noise_sim_start_s)}.")

        gain_error = np.exp(
            self.instrument.dets.gain_error * np.random.standard_normal(size=self.instrument.dets.n),
        )

        for field in self.loading:
            if field in ["noise"]:
                continue

            self.loading[field] *= gain_error[:, None]

    def __repr__(self):
        instrument_tree = "├ " + self.instrument.__repr__().replace("\n", "\n│ ")
        site_tree = "├ " + self.site.__repr__().replace("\n", "\n│ ")
        plan_tree = "├ " + self.plan.__repr__().replace("\n", "\n│ ")

        trees = []
        attrs = [attr for attr in ["atmosphere", "cmb", "map"] if getattr(self, attr, None)]
        for attr in attrs[:-1]:
            trees.append("├ " + getattr(self, attr).__repr__().replace("\n", "\n│ "))
        trees.append("└ " + getattr(self, attrs[-1]).__repr__().replace("\n", "\n  "))

        trees_string = "\n".join(trees)

        return f"""Simulation
{instrument_tree}
{site_tree}
{plan_tree}
{trees_string}"""

    @property
    def total_loading(self):
        return sum([d for d in self.loading.values()])
