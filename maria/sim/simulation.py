import logging
import os
import time as ttime

import numpy as np
from tqdm import tqdm

from ..atmosphere import Atmosphere
from ..base import BaseSimulation
from ..cmb import CMB, generate_cmb, get_cmb
from ..errors import PointingError
from ..instrument import Instrument
from ..map import Map
from ..plan import Plan
from ..site import Site
from .atmosphere import AtmosphereMixin
from .cmb import CMBMixin
from .map import MapMixin
from .noise import NoiseMixin

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

MIN_ELEVATION_WARN = 10
MIN_ELEVATION_ERROR = 5


class Simulation(BaseSimulation, AtmosphereMixin, CMBMixin, MapMixin, NoiseMixin):
    """
    A simulation of a telescope. This is what users should touch, primarily.
    """

    @classmethod
    def from_config(cls, config: dict = {}, **params):
        return cls(**params)

    def __init__(
        self,
        instrument: tuple[Instrument, str] = "default",
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
        verbose: bool = True,
    ):
        # self.parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params, strict=True)

        ref_time = ttime.monotonic()

        super().__init__(
            instrument,
            plan,
            site,
            verbose=verbose,
            # **self.parsed_sim_kwargs["instrument"],
            # **self.parsed_sim_kwargs["plan"],
            # **self
            # .parsed_sim_kwargs["site"],
        )

        duration = ttime.monotonic() - ref_time
        ref_time = ttime.monotonic()

        logger.info(f"Initialized base in {int(1e3 * duration)} ms.")

        self.noise = noise

        self.atmosphere_kwargs = atmosphere_kwargs
        self.cmb_kwargs = cmb_kwargs
        self.map_kwargs = map_kwargs
        self.noise_kwargs = noise_kwargs

        if map:
            self.map = map.to(units="K_RJ")

        if atmosphere:
            el_min = np.atleast_1d(self.coords.el).min().compute()
            if el_min < np.radians(MIN_ELEVATION_WARN):
                logger.warning(
                    f"Some detectors come within {MIN_ELEVATION_WARN} degrees of the horizon"
                    f"(el_min = {np.degrees(el_min):.01f}°)"
                )
            if el_min <= np.radians(MIN_ELEVATION_ERROR):
                raise PointingError(
                    f"Some detectors come within {MIN_ELEVATION_ERROR} degrees of the horizon"
                    f"(el_min = {np.degrees(el_min):.01f}°)"
                )

            weather_kwargs = (
                atmosphere_kwargs.pop("weather")
                if "weather" in atmosphere_kwargs
                else {}
            )

            ref_time = ttime.monotonic()

            self.atmosphere = Atmosphere(
                model=atmosphere,
                timestamp=self.plan.time.mean(),
                region=self.site.region,
                altitude=self.site.altitude,
                weather_kwargs=weather_kwargs,
            )

            # give it the simulation, so that it knows about pointing, site, etc.
            # kind of cursed
            self.atmosphere.initialize(self)

            duration = ttime.monotonic() - ref_time
            logger.info(f"Initialized atmosphere in {int(1e3 * duration)} ms.")
            ref_time = ttime.monotonic()

        if cmb:
            if cmb in ["spectrum", "power_spectrum", "generate", "generated"]:
                for _ in tqdm(range(1), desc="Generating CMB"):
                    self.cmb = generate_cmb(verbose=self.verbose, **cmb_kwargs)
            elif cmb in ["real", "planck"]:
                self.cmb = get_cmb(**cmb_kwargs)
            else:
                raise ValueError(f"Invalid value for cmb: '{cmb}'.")

    def _run(self):
        if hasattr(self, "atmosphere"):
            self._simulate_atmosphere()
            self._compute_atmospheric_emission()
            self._compute_atmospheric_transmission()

        if hasattr(self, "cmb"):
            self._simulate_cmb_emission()

        if hasattr(self, "map"):
            self._sample_maps()

        # number of bands are lost here
        if self.noise:
            self._simulate_noise()

        # scale the noise so that there is
        if hasattr(self, "atmospheric_transmission"):
            for k in ["cmb", "map"]:
                if k in self.data:
                    self.data[k] *= self.atmospheric_transmission

        gain_error = np.exp(
            self.instrument.dets.gain_error
            * np.random.standard_normal(size=self.instrument.dets.n)
        )

        for field in self.data:
            if field in ["noise"]:
                continue

            self.data[field] *= gain_error[:, None]

    def __repr__(self):
        # object_reprs = [
        #     getattr(self, attr).__repr__() for attr in ["instrument", "site", "plan"]
        # ]
        return "\n".join(
            [self.instrument.__repr__(), self.site.__repr__(), self.plan.__repr__()]
        )
