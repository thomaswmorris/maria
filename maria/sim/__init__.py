import os
import time as ttime

from tqdm import tqdm

from ..atmosphere import Atmosphere
from ..atmosphere.sim import AtmosphereMixin
from ..cmb import CMB, CMBMixin, generate_cmb, get_cmb
from ..instrument import Instrument
from ..map import Map, MapMixin
from ..noise import NoiseMixin
from ..plan import Plan
from ..site import Site
from .base import BaseSimulation

here, this_filename = os.path.split(__file__)


class Simulation(BaseSimulation, AtmosphereMixin, CMBMixin, MapMixin, NoiseMixin):
    """A simulation of a telescope. This is what users should touch, primarily."""

    @classmethod
    def from_config(cls, config: dict = {}, **params):
        return cls(**params)

    def __init__(
        self,
        instrument: Instrument or str = "default",
        plan: Plan or str = "daisy",
        site: Site or str = "hoagie_haven",
        atmosphere: str = None,
        cmb: CMB or str = None,
        map: Map = None,
        noise: bool = True,
        atmosphere_kwargs: dict = {},
        cmb_kwargs: dict = {},
        map_kwargs: dict = {},
        noise_kwargs: dict = {},
        verbose: bool = True,
    ):
        # self.parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params, strict=True)

        ref_time = ttime.time()

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

        duration = ttime.time() - ref_time
        ref_time = ttime.time()

        if self.verbose:
            print(f"Initialized base in {int(1e3 * duration)} ms.")

        self.noise = noise

        self.atmosphere_kwargs = atmosphere_kwargs
        self.cmb_kwargs = cmb_kwargs
        self.map_kwargs = map_kwargs
        self.noise_kwargs = noise_kwargs

        if map:
            self.map = map.to(units="K_RJ")

        if atmosphere:
            weather_kwargs = (
                atmosphere_kwargs.pop("weather")
                if "weather" in atmosphere_kwargs
                else {}
            )

            self.atmosphere = Atmosphere(
                t=self.plan.time.mean(),
                region=self.site.region,
                altitude=self.site.altitude,
                weather_kwargs=weather_kwargs,
            )

            if atmosphere == "2d":
                self._initialize_2d_atmosphere(**atmosphere_kwargs)

        if cmb:
            if cmb in ["spectrum", "power_spectrum", "generate", "generated"]:
                for _ in tqdm(range(1), desc="Generating CMB"):
                    self.cmb = generate_cmb(verbose=self.verbose, **cmb_kwargs)
            elif cmb in ["real", "planck"]:
                self.cmb = get_cmb(**cmb_kwargs)
            else:
                raise ValueError(f"Invalid value for cmb: '{cmb}'.")

    def _run(self):
        # number of bands are lost here
        if self.noise:
            self._simulate_noise()

        if hasattr(self, "atmosphere"):
            self._simulate_atmospheric_emission()

        if hasattr(self, "cmb"):
            self._simulate_cmb_emission()

        if hasattr(self, "map"):
            self._sample_maps()

        # scale the noise so that there is
        if hasattr(self, "atmospheric_transmission"):
            for k in ["cmb", "map"]:
                if k in self.data:
                    self.data[k] *= self.atmospheric_transmission

    def __repr__(self):
        object_reprs = [
            getattr(self, attr).__repr__() for attr in ["instrument", "site", "plan"]
        ]
        return ", ".join(object_reprs)
