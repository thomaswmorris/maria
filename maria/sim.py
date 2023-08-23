import glob, os

import numpy as np
from astropy.io import fits

from .tod import TOD
from .base import BaseSimulation

from .array import get_array, get_array_config
from .pointing import get_pointing, get_pointing_config
from .site import get_site, get_site_config

from . import atmosphere, cmb, noise, sky, utils

here, this_filename = os.path.split(__file__)

class InvalidSimulationParameterError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(f"The parameters {invalid_keys} are not valid simulation parameters!")

# valid sim kwargs are a kwarg in some config file
config_files = glob.glob(f"{here}/configs/*.yml")
VALID_SIM_KWARGS = set()
for fp in config_files:
    configs = utils.read_yaml(fp)
    for key, config in configs.items():
        VALID_SIM_KWARGS |= set(config.keys())

def _validate_kwargs(kwargs):
    invalid_keys = [key for key in kwargs.keys() if key not in VALID_SIM_KWARGS]
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
                 noise_model=None, 
                 **kwargs):

        #_validate_kwargs(kwargs)

        if isinstance(array, str):
            array = get_array(array, **kwargs)

        if isinstance(pointing, str):
            pointing = get_pointing(pointing, **kwargs)

        if isinstance(site, str):
            site = get_site(site, **kwargs)

        super().__init__(array, pointing, site)

        self.atm_model = atm_model
        if atm_model in ["linear_angular", "LA"]:
            self.atm_sim = atmosphere.LinearAngularSimulation(array, pointing, site, **kwargs)
        elif atm_model in ["kolmogorov_taylor", "KT"]:
            self.atm_sim = atmosphere.KolmogorovTaylorSimulation(array, pointing, site, **kwargs)
        else:
            self.atm_sim = None

        self.map_file = map_file
        if map_file is not None:
            self.map_sim = sky.MapSimulation(array, pointing, site, map_file, **kwargs)
        else:
            self.map_sim = None

    def run(self):

        if self.atm_sim is not None:
            self.atm_sim.run()

        if self.map_sim is not None:
            self.map_sim.run()

        tod = TOD()

        tod.time = self.pointing.time
        tod.az   = self.pointing.az
        tod.el   = self.pointing.el
        tod.ra   = self.pointing.ra
        tod.dec  = self.pointing.dec

        # number of bands are lost here
        tod.data = np.zeros((self.array.n_dets, self.pointing.n_time))

        if self.atm_sim is not None:
            tod.data += self.atm_sim.temperature

        if self.map_sim is not None:
            tod.data += self.map_sim.temperature

        tod.dets = self.array.dets

        tod.meta = {'latitude': self.site.latitude,
                    'longitude': self.site.longitude,
                    'altitude': self.site.altitude}

        return tod

