import os

from maria.instrument import Instrument
from maria.plan import Plan
from maria.site import Site

from ..atmosphere import Atmosphere, AtmosphereMixin
from ..cmb import CMBMixin
from ..map import MapMixin
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

        self.noise = True

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

        if self.map_file:
            if not os.path.exists(self.map_file):
                raise FileNotFoundError(self.map_file)
            self._initialize_map()

        if self.atmosphere_model:
            weather_override = {k: v for k, v in {"pwv": self.pwv}.items() if v}

            self.atmosphere = Atmosphere(
                t=self.plan.time.mean(),
                region=self.site.region,
                altitude=self.site.altitude,
                weather_override=weather_override,
            )

            self._initialize_atmosphere()

        if self.cmb_source:
            self._initialize_cmb()

    def _run(self):
        # number of bands are lost here
        if self.noise:
            self._simulate_noise()

        if hasattr(self, "atmosphere"):
            self._simulate_atmospheric_emission()

        if hasattr(self, "cmb"):
            self._simulate_cmb_emission()

            # convert to source Rayleigh-Jeans
            # self.data["atmosphere"] *= self.instrument.dets.abs_cal_rj[:, None]
            # self.data["atmosphere"] /= self.atmospheric_transmission

        if hasattr(self, "map"):
            self._sample_maps()

        self.tod_data = sum(
            [self.data.get(k, 0) for k in ["atmosphere", "cmb", "map", "noise"]]
        )

        if hasattr(self, "atmospheric_transmission"):
            self.tod_data /= self.atmospheric_transmission

        self.tod_data *= self.instrument.dets.abs_cal_rj[:, None]

    def __repr__(self):
        object_reprs = [
            getattr(self, attr).__repr__() for attr in ["instrument", "site", "plan"]
        ]
        return ", ".join(object_reprs)
