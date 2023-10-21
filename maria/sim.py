import os

import numpy as np

from . import atmosphere, noise, sky, utils
from .base import BaseSimulation, parse_sim_kwargs

from .array import ARRAY_PARAMS, get_array
from .pointing import POINTING_PARAMS, get_pointing
from .site import SITE_PARAMS, get_site

from .atmosphere import ATMOSPHERE_PARAMS
from .sky import MAP_PARAMS

here, this_filename = os.path.split(__file__)

from maria.array import Array
from maria.pointing import Pointing
from maria.site import Site

master_params = utils.io.read_yaml(f"{here}/params.yml")

class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(
            f"The parameters {invalid_keys} are not valid simulation parameters!"
        )

class Simulation(BaseSimulation):
    """A simulation! This is what users should touch, primarily.
    """

    def __init__(
        self,
        array: str or Array,
        pointing: str or Pointing,
        site: str or Site,
        atm_model=None,
        noise_model=None,
        map_file=None,
        map_center=None,
        verbose=False,
        **kwargs,
    ):

        parsed_sim_kwargs = parse_sim_kwargs(kwargs, master_params, strict=True)

        super().__init__(array, pointing, site, **parsed_sim_kwargs["array"], **parsed_sim_kwargs["pointing"], **parsed_sim_kwargs["site"])

        self.atm_sim = None
        self.map_sim = None
        self.noise_sim = None

        if atm_model is not None:
            self.atm_model = None
            atm_kwargs = parsed_sim_kwargs["atmosphere"]
            if atm_model in ["single_layer", "SL"]:
                self.atm_sim = atmosphere.SingleLayerSimulation(
                    self.array, self.pointing, self.site, **atm_kwargs
                )
            elif atm_model in ["kolmogorov_taylor", "KT"]:
                self.atm_sim = atmosphere.KolmogorovTaylorSimulation(
                    self.array, self.pointing, self.site, **atm_kwargs
                )

        if map_file is not None:
            self.map_file = map_file
            map_kwargs = parsed_sim_kwargs["map"]
            if map_file is not None:
                self.map_sim = sky.MapSimulation(self.array, self.pointing, self.site, map_file, **map_kwargs)


    def _run(self):

        # number of bands are lost here
        self.data = np.zeros((self.array.n_dets, self.pointing.n_time))

        if self.atm_sim is not None:
            self.atm_sim._run()
            self.data += self.atm_sim.data

        if self.map_sim is not None:
            self.map_sim._run()
            self.data += self.map_sim.data