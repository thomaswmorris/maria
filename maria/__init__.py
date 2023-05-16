# Ave, María, grátia plena, Dóminus tecum

from ._version import get_versions
__version__ = get_versions()["version"]

del get_versions

import os
import numpy as np
import scipy as sp

import astropy as ap

import pandas as pd
import h5py
import glob
import re
import json
import time
import copy

import weathergen
from tqdm import tqdm

import warnings
import healpy as hp

from matplotlib import pyplot as plt
from astropy.io import fits

here, this_filename = os.path.split(__file__)

from . import atmosphere, sky, simulations
from .objects import Array, Pointing, Site

with open(f'{here}/configs/arrays.json', 'r+') as f:
    ARRAY_CONFIGS = json.load(f)

with open(f'{here}/configs/pointings.json', 'r+') as f:
    POINTING_CONFIGS = json.load(f)

with open(f'{here}/configs/sites.json', 'r+') as f:
    SITE_CONFIGS = json.load(f)

ARRAYS = list((ARRAY_CONFIGS.keys()))
POINTINGS = list((POINTING_CONFIGS.keys()))
SITES = list((SITE_CONFIGS.keys()))

class InvalidArrayError(Exception):
    def __init__(self, invalid_array):
        super().__init__(f"The array \'{invalid_array}\' is not in the database of default arrays. "
        f"Default arrays are:\n\n{sorted(list(ARRAY_CONFIGS.keys()))}")

class InvalidPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(f"The site \'{invalid_pointing}\' is not in the database of default pointings. "
        f"Default pointings are:\n\n{sorted(list(POINTING_CONFIGS.keys()))}")

class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(f"The site \'{invalid_site}\' is not in the database of default sites. "
        f"Default sites are:\n\n{sorted(list(SITE_CONFIGS.keys()))}")

class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        region_string = regions.to_string(columns=['location', 'country', 'latitude', 'longitude', 'altitude'])
        super().__init__(f"The region \'{invalid_region}\' is not supported. Supported regions are:\n\n{region_string}")


def get_array_config(array_name, **kwargs):
    if not array_name in ARRAY_CONFIGS.keys():
        raise InvalidArrayError(array_name)
    ARRAY_CONFIG = ARRAY_CONFIGS[array_name].copy()
    for k, v in kwargs.items():
        ARRAY_CONFIG[k] = v
    return ARRAY_CONFIG


def get_pointing_config(pointing_name, **kwargs):
    if not pointing_name in POINTING_CONFIGS.keys():
        raise InvalidPointingError(pointing_name)
    POINTING_CONFIG = POINTING_CONFIGS[pointing_name].copy()
    for k, v in kwargs.items():
        POINTING_CONFIG[k] = v
    return POINTING_CONFIG


def get_site_config(site_name, **kwargs):
    if not site_name in SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    SITE_CONFIG = SITE_CONFIGS[site_name].copy()
    for k, v in kwargs.items():
        SITE_CONFIG[k] = v
    return SITE_CONFIG

def get_array(array_name, **kwargs):
    return Array(**get_array_config(array_name, **kwargs))

def get_pointing(pointing_name, **kwargs):
    return Pointing(**get_pointing_config(pointing_name, **kwargs))

def get_site(site_name, **kwargs):
    return Site(**get_site_config(site_name, **kwargs))



class TOD:

    def __init__(self):
        pass

    def subset(self, mask):

        tod_subset = copy.deepcopy(self)

        tod_subset.data      = tod_subset.data[mask]
        tod_subset.detectors = tod_subset.detectors.loc[mask]

        return tod_subset


    
    def plot(self):
        pass


class Simulation(simulations.BaseSimulation):
    """
    A simulation! This is what users should touch, primarily. 
    """
    def __init__(self, array, pointing, site, atm_model="linear_angular", noise_model=None):
        super().__init__(array, pointing, site)

        if atm_model == "linear_angular":
            self.atm_sim = atmosphere.LinearAngularSimulation(array, pointing, site)
        else:
            self.atm_sim = None

    def run(self):

        if self.atm_sim is not None:

            self.atm_sim.run()

        tod = TOD()

        tod.time = self.pointing.unix
        tod.az   = self.pointing.az
        tod.el   = self.pointing.el
        tod.ra   = self.pointing.ra
        tod.dec  = self.pointing.dec

        tod.data = np.zeros((self.array.n_det, self.pointing.n_time))

        tod.data += self.atm_sim.temperature

        tod.detectors = self.array.metadata

        tod.metadata = {'latitude': self.site.latitude,
                        'longitude': self.site.longitude,
                        'altitude': self.site.altitude}

        return tod
