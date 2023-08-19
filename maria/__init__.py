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

REGIONS_WITH_SPECTRA = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
REGIONS_WITH_WEATHER = list(weathergen.regions.index)

SUPPORTED_REGIONS = list(set(REGIONS_WITH_SPECTRA) & set(REGIONS_WITH_WEATHER))

regions = weathergen.regions.loc[SUPPORTED_REGIONS].sort_index()

from . import base, atmosphere, sky, noise
from .objects import Array, Pointing, Site

here, this_filename = os.path.split(__file__)


ARRAY_CONFIGS = utils.read_yaml(f'{here}/configs/arrays.yml')
POINTING_CONFIGS = utils.read_yaml(f'{here}/configs/pointings.yml')
SITE_CONFIGS = utils.read_yaml(f'{here}/configs/sites.yml')

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


    def to_fits(self, filename):

        """
        """

        ...


    def from_fits(self, filename):

        """
        """

        ...

        
    def to_hdf(self, filename):

        with h5py.File(filename, "w") as f:

            f.create_dataset("")
    
    def plot(self):
        pass

class KeyNotFoundError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(f"The key \'{invalid_keys}\' is not in the database. ")

def check_nested_keys(keys_found, data, keys):

    for key in data.keys():
        for i in range(len(keys)):
            if keys[i] in data[key].keys():
                keys_found[i] = True

def check_json_file_for_key(keys_found, file_path, *keys_to_check):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)       
        return check_nested_keys(keys_found, data, keys_to_check)

def test_multiple_json_files(files_to_test, *keys_to_find):   
    keys_found = np.zeros(len(keys_to_find)).astype(bool)

    for file_path in files_to_test:
        check_json_file_for_key(keys_found, file_path, *keys_to_find)

    if np.sum(keys_found) != len(keys_found):
        raise KeyNotFoundError(np.array(keys_to_find)[~keys_found])

class Simulation(base.BaseSimulation):
    """
    A simulation! This is what users should touch, primarily. 
    """
    def __init__(self, 
                 array, 
                 pointing, 
                 site, 
                 atm_model="linear_angular", 
                 map_file=None, 
                 noise_model="white", 
                 **kwargs):

        if isinstance(array, str):
            array = get_array(array, **kwargs)

        if isinstance(pointing, str):
            pointing = get_pointing(pointing, **kwargs)

        if isinstance(site, str):
            site = get_site(site, **kwargs)

        json_pattern = os.path.join(here+'/configs', '*.json')
        json_files    = glob.glob(json_pattern)
        test_multiple_json_files(json_files, *tuple(kwargs.keys()))

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

        tod.time = self.pointing.unix
        tod.az   = self.pointing.az
        tod.el   = self.pointing.el
        tod.ra   = self.pointing.ra
        tod.dec  = self.pointing.dec

        # number of bands are lost here
        tod.data = np.zeros((self.array.n_det, self.pointing.n_time))

        if self.atm_sim is not None:
            tod.data += self.atm_sim.temperature

        if self.map_sim is not None:
            tod.data += self.map_sim.temperature

        tod.detectors = self.array.metadata

        tod.metadata = {'latitude': self.site.latitude,
                        'longitude': self.site.longitude,
                        'altitude': self.site.altitude}

        return tod

    def save_maps(self, mapper):

            self.map_sim.map_data["header"]['comment'] = 'Made Synthetic observations via maria code'
            self.map_sim.map_data["header"]['comment'] = 'Changed resolution and size of the output map'
            self.map_sim.map_data["header"]['CDELT1']  = np.rad2deg(mapper.resolution)
            self.map_sim.map_data["header"]['CDELT2']  = np.rad2deg(mapper.resolution)
            self.map_sim.map_data["header"]['CRPIX1']  = mapper.maps[list(mapper.maps.keys())[0]].shape[0]
            self.map_sim.map_data["header"]['CRPIX2']  = mapper.maps[list(mapper.maps.keys())[0]].shape[1]

            self.map_sim.map_data["header"]['comment'] = 'Changed pointing location of the output map'
            self.map_sim.map_data["header"]['CRVAL1']  = self.pointing.ra.mean()
            self.map_sim.map_data["header"]['CRVAL2']  = self.pointing.dec.mean()

            self.map_sim.map_data["header"]['comment'] = 'Changed spectral position of the output map'
            self.map_sim.map_data["header"]['CTYPE3']  = 'FREQ    '
            self.map_sim.map_data["header"]['CUNIT3']  = 'Hz      '
            self.map_sim.map_data["header"]['CRPIX3']  = 1.000000000000E+00
            
            self.map_sim.map_data["header"]['BTYPE']   = 'Intensity'
            if self.map_sim.map_data["inbright"] == 'Jy/pixel': 
                self.map_sim.map_data["header"]['BUNIT']   = 'Jy/beam '   
            else: 
                self.map_sim.map_data["header"]['BUNIT']   = 'Kelvin RJ'   

            for i in range(len(self.array.ubands)):
                
                self.map_sim.map_data["header"]['CRVAL3']  = self.array.detectors[i][0]
                self.map_sim.map_data["header"]['CDELT3']  = self.array.detectors[i][1]

                save_map = mapper.maps[list(mapper.maps.keys())[i]] 

                if self.map_sim.map_data["inbright"] == 'Jy/pixel': 
                    save_map *= utils.KbrightToJyPix(self.map_data["header"]['CRVAL3'], 
                                                     self.map_data["header"]['CDELT1'], 
                                                     self.map_data["header"]['CDELT2']
                                                    )
                fits.writeto( filename = here+'/outputs/atm_model_'+str(self.atm_model)+'_file_'+ str(self.map_file.split('/')[-1].split('.fits')[0])+'.fits', 
                                  data = save_map, 
                                header = self.map_sim.map_data["header"],
                             overwrite = True 
                            )

