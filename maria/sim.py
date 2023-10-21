import glob, os
from . import atmosphere

import numpy as np
from astropy.io import fits

from .base import BaseSimulation

from .array import get_array, get_array_config
from .pointing import get_pointing, get_pointing_config
from .site import get_site, get_site_config

from . import cmb, noise, sky, utils

here, this_filename = os.path.split(__file__)

here, this_filename = os.path.split(__file__)

from .array import get_array, ARRAY_PARAMS
from .coordinator import Coordinator
from .pointing import get_pointing, POINTING_PARAMS
from .site import get_site, SITE_PARAMS
from .sky import MAP_PARAMS
from .tod import TOD

from . import utils

VALID_SIM_PARAMS = ARRAY_PARAMS | POINTING_PARAMS | SITE_PARAMS | MAP_PARAMS

class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(f"The parameters {invalid_keys} are not valid simulation parameters!")

def _validate_kwargs(kwargs):
    invalid_keys = [key for key in kwargs.keys() if key not in VALID_SIM_PARAMS]
    if len(invalid_keys) > 0:
        raise InvalidSimulationParameterError(invalid_keys)

class Simulation(BaseSimulation):
    """
    A simulation! This is what users should touch, primarily. 
    """
    def __init__(self, 
                 array, 
                 pointing, 
                 site, 
                 atm_model=None, 
                 map_file=None, 
                 map_center=None,
                 noise_model=None,
                 **kwargs):

        super().__init__(array, pointing, site, **kwargs)

        self.atm_model = atm_model
        if atm_model in ["single_layer", "SL"]:
            self.atm_sim = atmosphere.SingleLayerSimulation(array, pointing, site, **kwargs)
        elif atm_model in ["kolmogorov_taylor", "KT"]:
            self.atm_sim = atmosphere.KolmogorovTaylorSimulation(array, pointing, site, **kwargs)
        else:
            self.atm_sim = None

        self.map_file = map_file
        if map_file is not None:
            self.map_sim = sky.MapSimulation(array, pointing, site, map_file, **kwargs)
        else:
            self.map_sim = None


    def _run(self):

        if self.atm_sim is not None:
            self.atm_sim._run()

        if self.map_sim is not None:
            self.map_sim._run()

        # number of bands are lost here
        self.data = np.zeros((self.array.n_dets, self.pointing.n_time))

        if self.atm_sim is not None:
            self.data += self.atm_sim.data

        if self.map_sim is not None:
            self.data += self.map_sim.data