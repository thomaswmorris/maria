from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import numpy as np
from tqdm import tqdm

from ..atmosphere import DEFAULT_ATMOSPHERE_KWARGS, Atmosphere
from ..cmb import CMB, DEFAULT_CMB_KWARGS, generate_cmb, get_cmb
from ..coords import Coordinates
from ..errors import PointingError
from ..instrument import Instrument, get_instrument
from ..io import fetch, humanize_time
from ..map import Map, load
from ..plan import Plan, PlanList, get_plan
from ..site import Site, get_site
from ..tod import TOD
from ..utils import read_yaml
from .atmosphere import AtmosphereMixin
from .base import BaseSimulation
from .cmb import CMBMixin
from .map import MapMixin
from .noise import NoiseMixin
from .observation import Observation

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

MIN_ELEVATION_WARN = 10  # degrees
MIN_ELEVATION_ERROR = 5  # degrees


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!",
        )


master_params = read_yaml(f"{here}/params.yml")


def parse_sim_kwargs(kwargs, master_kwargs, strict=False):
    parsed_kwargs = {k: {} for k in master_kwargs.keys()}
    invalid_kwargs = {}

    for k, v in kwargs.items():
        parsed = False
        for sub_type, sub_kwargs in master_kwargs.items():
            if k in sub_kwargs.keys():
                parsed_kwargs[sub_type][k] = v
                parsed = True
        if not parsed:
            invalid_kwargs[k] = v

    if len(invalid_kwargs) > 0:
        if strict:
            raise InvalidSimulationParameterError(
                invalid_keys=list(invalid_kwargs.keys()),
            )

    return parsed_kwargs


class Simulation(AtmosphereMixin, CMBMixin, MapMixin, NoiseMixin):
    """
    A simulation of a telescope. This is what users should touch, primarily.
    """

    @classmethod
    def from_config(cls, config: dict = {}, **params):
        return cls(**{**config, **params})

    def __init__(
        self,
        instrument: Instrument | str,
        plans: PlanList | list[Plan | str],
        site: Site | str,
        atmosphere: Atmosphere | str = None,
        atmosphere_kwargs: dict = {},
        cmb: CMB | str = None,
        cmb_kwargs: dict = {},
        map: Map | str = None,
        map_kwargs: dict = {},
        noise: bool = True,
        noise_kwargs: dict = {},
        progress_bars: bool = True,
        keep_mean_signal: bool = False,
        dtype: type = np.float32,
    ):
        self.atmosphere = atmosphere
        self.atmosphere_kwargs = DEFAULT_ATMOSPHERE_KWARGS.copy()
        self.atmosphere_kwargs.update(atmosphere_kwargs)

        self.map_kwargs = map_kwargs

        self.noise = noise
        self.noise_kwargs = noise_kwargs

        self.dtype = dtype
        self.disable_progress_bars = (not progress_bars) or (logging.getLevelName(logger.level) == "DEBUG")

        instrument_init_s = ttime.monotonic()

        # parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params)

        if isinstance(instrument, Instrument):
            self.instrument = instrument
        else:
            self.instrument = get_instrument(name=instrument)

        logger.debug(f"Initialized instrument in {humanize_time(ttime.monotonic() - instrument_init_s)}.")
        site_init_s = ttime.monotonic()

        if isinstance(site, Site):
            self.site = site
        elif isinstance(site, str):
            self.site = get_site(site_name=site)
        else:
            raise ValueError(
                "'site' must be either a Site object or a string.",
            )

        logger.debug(f"Initialized site in {humanize_time(ttime.monotonic() - site_init_s)}.")
        plan_init_s = ttime.monotonic()

        if isinstance(plans, str):
            plans = [get_plan(plan_name=plans)]
        elif isinstance(plans, Plan):
            plans = [plans]
        elif isinstance(plans, PlanList):
            plans = plans
        elif not isinstance(plans, list):
            raise TypeError("plans must be a plan or a list of plans")

        self.plans = PlanList(plans)

        logger.debug(f"Initialized plans in {humanize_time(ttime.monotonic() - plan_init_s)}.")

        self.obs_list = []
        for plan in self.plans:
            obs_weather_kwargs = self.atmosphere_kwargs.pop("weather") if "weather" in self.atmosphere_kwargs else {}
            obs = Observation(
                instrument=self.instrument,
                plan=plan,
                site=self.site,
                atmosphere=self.atmosphere,
                atmosphere_kwargs=self.atmosphere_kwargs,
                weather_kwargs=obs_weather_kwargs,
            )

            self.obs_list.append(obs)

        if cmb:
            cmb_start_s = ttime.monotonic()
            self._init_cmb(cmb, **cmb_kwargs)
            logger.debug(f"Initialized CMB simulation in {humanize_time(ttime.monotonic() - cmb_start_s)}.")

        if map:
            map_start_s = ttime.monotonic()
            self._init_map(map, **map_kwargs)
            logger.debug(f"Initialized map simulation in {humanize_time(ttime.monotonic() - map_start_s)}.")

        if noise:
            noise_start_s = ttime.monotonic()
            logger.debug(f"Initialized noise simulation in {humanize_time(ttime.monotonic() - noise_start_s)}.")

        #     # self.start = arrow.get(self.boresight.t.min()).to("utc")
        #     # self.end = arrow.get(self.boresight.t.max()).to("utc")

        #     if atmosphere:
        #         atmosphere_init_start_s = ttime.monotonic()

        #         # give it the observation, so that it knows about pointing, site, etc. (kind of cursed)
        #         obs.atmosphere.initialize(obs)

        #         logger.debug(
        #             f"Initialized atmosphere simulation in {humanize_time(ttime.monotonic() - atmosphere_init_start_s)}."
        #         )

        # logger.debug(f"Initialized simulation in {humanize_time(ttime.monotonic() - sim_start_s)}.")

    def run(self, units: str = "K_RJ"):
        tods = []

        for obs_index, obs in enumerate(self.obs_list):
            obs_start_s = ttime.monotonic()

            logger.info(f"Simulating observation {obs_index + 1} of {len(self.obs_list)}")

            obs.loading = {}

            if hasattr(obs, "atmosphere"):
                atmosphere_sim_start_s = ttime.monotonic()
                obs.atmosphere.initialize(instrument=obs.instrument, boresight=obs.boresight, site=obs.site)
                self._simulate_atmosphere(obs)
                obs.loading["atmosphere"] = self._compute_atmospheric_loading(obs)
                logger.debug(f"Ran atmosphere simulation in {humanize_time(ttime.monotonic() - atmosphere_sim_start_s)}.")

            if hasattr(self, "cmb"):
                cmb_sim_start_s = ttime.monotonic()
                obs.loading["cmb"] = self._compute_cmb_loading(obs)
                logger.debug(f"Ran CMB simulation in {humanize_time(ttime.monotonic() - cmb_sim_start_s)}.")

            if hasattr(self, "map"):
                map_sim_start_s = ttime.monotonic()
                self._sample_maps(obs)
                logger.debug(f"Ran map simulation in {humanize_time(ttime.monotonic() - map_sim_start_s)}.")

            # number of bands are lost here
            if self.noise:
                noise_sim_start_s = ttime.monotonic()
                self._simulate_noise(obs)
                logger.debug(f"Ran noise simulation in {humanize_time(ttime.monotonic() - noise_sim_start_s)}.")

            gain_error = np.exp(
                obs.instrument.dets.gain_error * np.random.standard_normal(size=obs.instrument.dets.n),
            )

            for field in obs.loading:
                if field in ["noise"]:
                    continue

                obs.loading[field] *= gain_error[:, None]

            metadata = {
                "atmosphere": False,
                "sim_time": arrow.now(),
                "altitude": float(obs.site.altitude.m),
                "region": obs.site.region,
            }

            if hasattr(obs, "atmosphere"):
                metadata["atmosphere"] = True
                metadata["pwv"] = float(np.round(obs.atmosphere.weather.pwv, 3))
                metadata["base_temperature"] = float(np.round(obs.atmosphere.weather.temperature[0], 3))
            else:
                metadata["atmosphere"] = False

            tod = TOD(
                data=obs.loading,
                dets=obs.instrument.dets,
                coords=obs.coords,
                units="pW",
                metadata=metadata,
            )

            tods.append(tod.to(units))

            logger.info(
                f"Simulated observation {obs_index + 1} of {len(self.obs_list)} "
                f"in {humanize_time(ttime.monotonic() - obs_start_s)}"
            )

        return tods

    def plot_counts(self, x_bins=100, y_bins=100):
        self.plan.plot_counts(instrument=self.instrument, x_bins=x_bins, y_bins=y_bins)

    def __repr__(self):
        instrument_tree = "├ " + self.instrument.__repr__().replace("\n", "\n│ ")
        site_tree = "├ " + self.site.__repr__().replace("\n", "\n│ ")
        plan_tree = "├ " + self.plans.__repr__().replace("\n", "\n│ ")

        trees = []
        attrs = [attr for attr in ["atmosphere", "cmb", "map"] if getattr(self, attr, None)]
        if attrs:
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

    @property
    def min_time(self):
        return self.obs_list[0].plan.start_time

    @property
    def max_time(self):
        return self.obs_list[-1].plan.end_time
