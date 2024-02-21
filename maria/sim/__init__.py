import os

from maria.instrument import Instrument
from maria.plan import Plan
from maria.site import Site

from ..atmosphere import Atmosphere, AtmosphereMixin
from ..cmb import CMBMixin
from ..map import Map, MapMixin
from ..noise import NoiseMixin
from .base import BaseSimulation, master_params, parse_sim_kwargs

here, this_filename = os.path.split(__file__)


class Simulation(BaseSimulation, AtmosphereMixin, CMBMixin, MapMixin, NoiseMixin):
    """A simulation! This is what users should touch, primarily."""

    @classmethod
    def from_config(cls, config: dict = {}, **params):
        return cls(**params)

    def __init__(
        self,
        instrument: str or Instrument = "MUSTANG-2",
        plan: str or Plan = "stare",
        site: str or Site = "hoagie_haven",
        map: str or Map = None,
        verbose: bool = True,
        **kwargs,
    ):
        self.parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params, strict=True)

        super().__init__(
            instrument,
            plan,
            site,
            verbose=verbose,
            **self.parsed_sim_kwargs["instrument"],
            **self.parsed_sim_kwargs["plan"],
            **self.parsed_sim_kwargs["site"],
        )

        self.params = {}

        for sub_type, sub_master_params in master_params.items():
            self.params[sub_type] = {}
            if sub_type in ["instrument", "site", "plan"]:
                sub_type_dataclass = getattr(self, sub_type)
                for k in sub_type_dataclass.__dataclass_fields__.keys():
                    v = getattr(sub_type_dataclass, k)
                    setattr(self, k, v)
                    self.params[sub_type][k] = v
            else:
                for k, v in sub_master_params.items():
                    setattr(self, k, kwargs.get(k, v))
                    self.params[sub_type][k] = v

        weather_override = {k: v for k, v in {"pwv": self.pwv}.items() if v}

        self.weather = Weather(
            t=self.plan.time.mean(),
            region=self.site.region,
            altitude=self.site.altitude,
            quantiles=self.site.weather_quantiles,
            override=weather_override,
        )

        if self.map_file:
            if not os.path.exists(self.map_file):
                raise FileNotFoundError(self.map_file)
            self._initialize_map()

        if self.atmosphere_model:
            self._initialize_atmosphere()

    def _run(self, units="K_RJ"):
        # number of bands are lost here
        self._simulate_noise()

        if self.atmosphere_model:
            self._simulate_atmospheric_emission()

        if self.map_file:
            self._sample_maps()

            if hasattr(self, "atmospheric_transmission"):
                self.data["map"] *= self.atmospheric_transmission

        if hasattr(self, "cmb_sim"):
            self.cmb_sim._run()
            self.data += self.cmb_sim.data

    def __repr__(self):
        object_reprs = [
            getattr(self, attr).__repr__() for attr in ["instrument", "site", "plan"]
        ]
        return "\n\n".join(object_reprs)
