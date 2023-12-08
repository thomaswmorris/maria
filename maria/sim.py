import os

import numpy as np

from maria.array import Array
from maria.pointing import Pointing
from maria.site import Site

from . import atmosphere, noise, sky, utils
from .base import BaseSimulation, parse_sim_kwargs

here, this_filename = os.path.split(__file__)

master_params = utils.io.read_yaml(f"{here}/configs/params.yml")


class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!"
        )


class Simulation(BaseSimulation):
    """A simulation! This is what users should touch, primarily."""

    def __init__(
        self,
        array: str or Array,
        pointing: str or Pointing,
        site: str or Site,
        white_noise_model="white",
        pink_noise_model="pink",
        atm_model=None,
        verbose=False,
        **kwargs,
    ):
        self.parsed_kwargs = parse_sim_kwargs(kwargs, master_params, strict=True)

        super().__init__(
            array,
            pointing,
            site,
            verbose=verbose,
            **self.parsed_kwargs["array"],
            **self.parsed_kwargs["pointing"],
            **self.parsed_kwargs["site"],
        )

        self.atm_sim = None
        self.map_sim = None
        self.white_noise_sim = None
        self.pink_noise_sim = None

        self.sub_sims = []

        if atm_model is not None:
            atm_kwargs = self.parsed_kwargs["atmosphere"]
            if atm_model in ["single_layer", "SL"]:
                self.atm_sim = atmosphere.SingleLayerSimulation(
                    self.array, self.pointing, self.site, **atm_kwargs
                )
            elif atm_model in ["linear_angular", "LA"]:
                self.atm_sim = atmosphere.LinearAngularSimulation(
                    self.array, self.pointing, self.site, **atm_kwargs
                )
            elif atm_model in ["kolmogorov_taylor", "KT"]:
                self.atm_sim = atmosphere.KolmogorovTaylorSimulation(
                    self.array, self.pointing, self.site, **atm_kwargs
                )

            else:
                raise ValueError()

            self.sub_sims.append(self.atm_sim)

        if white_noise_model is not None:
            noise_kwargs = self.parsed_kwargs["noise"]
            if "white" in white_noise_model:
                self.white_noise_sim = noise.WhiteNoiseSimulation(
                    self.array, self.pointing, self.site, **noise_kwargs
                )
            else:
                raise ValueError()

            self.sub_sims.append(self.white_noise_sim)

        if pink_noise_model is not None:
            noise_kwargs = self.parsed_kwargs["noise"]
            if "pink" in pink_noise_model:
                self.pink_noise_sim = noise.PinkNoiseSimulation(
                    self.array, self.pointing, self.site, **noise_kwargs
                )
            else:
                raise ValueError()

            self.sub_sims.append(self.pink_noise_sim)

        if "map_file" in kwargs.keys():
            map_kwargs = self.parsed_kwargs["map"]
            self.map_sim = sky.MapSimulation(
                self.array, self.pointing, self.site, **map_kwargs
            )

            self.sub_sims.append(self.map_sim)

    def _run(self, units="K_RJ"):
        # number of bands are lost here
        self.data = np.zeros((self.array.n_dets, self.pointing.n_time))

        for sim in self.sub_sims:
            sim.run()
            self.data += sim.data
