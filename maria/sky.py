import numpy as np
import scipy as sp
import pandas as pd
import h5py
import os
from tqdm import tqdm
import warnings
from importlib import resources
import time as ttime
from . import utils
import weathergen
from os import path
import matplotlib.pyplot as plt
from datetime import datetime

from astropy.io import fits

here, this_filename = os.path.split(__file__)

from . import base

class InvalidNBandsError(Exception):
    def __init__(self, invalid_nbands):
        super().__init__(f"Number of bands  \'{invalid_nbands}\' don't match the cube size."
        f"The input fits file must be an image or a cube that match the number of bands")

class BaseSkySimulation(base.BaseSimulation):
    """
    This simulates scanning over celestial sources.
    """
    def __init__(self, array, pointing, site, **kwargs):
        super().__init__(array, pointing, site)

        AZIM, ELEV = utils.from_xy(
            self.array.offset_x[:, None],
            self.array.offset_y[:, None],
            self.pointing.az,
            self.pointing.el,
        )

        self.RA, self.DEC = self.coordinator.transform(
            self.pointing.unix,
            AZIM, ELEV,
            in_frame="az_el",
            out_frame="ra_dec",
        )

        self.X, self.Y = utils.to_xy(self.RA, self.DEC, self.RA.mean(), self.DEC.mean())

class MapSimulation(BaseSkySimulation):
    """
    This simulates scanning over celestial sources.
    """
    def __init__(self, 
                 array, 
                 pointing, 
                 site, 
                 map_file, 
                 **kwargs):

        super().__init__(array, pointing, site, **kwargs)

        # Prep mapmaker
        # -------------
        self.map_file = map_file
        hudl          = fits.open(self.map_file)
        map_image     = hudl[0].data
        *_, map_nx, map_ny = map_image.shape
        map_image = map_image.reshape(-1, map_nx, map_ny)

        if len(self.array.detectors) != map_image.shape[0] | map_image.shape[0] != 1:
            raise InvalidNBandsError(len(self.array.detectors))

        if len(self.array.detectors) != 1 & map_image.shape[0] == 1:
            _ = np.empty((len(self.array.detectors), map_nx, map_ny))
            for i in range(len(self.array.detectors)): _[i] = map_image[0]
            map_image = np.copy(_)

        # map is a degree wide by default
        DEFAULT_MAP_CENTER = (self.RA.mean(), self.DEC.mean())
        DEFAULT_MAP_RES = 1 / np.maximum(map_nx, map_ny)

        map_center   = kwargs.get("map_center",     DEFAULT_MAP_CENTER)
        map_res      = kwargs.get("map_res",        DEFAULT_MAP_RES)
        map_units    = kwargs.get("map_units",     'KRJ')
        map_inbright = kwargs.get("map_inbright",   None)
        
        map_header = hudl[0].header
        map_header['HISTORY'] = 'History_input_adjustments'
        map_header['comment'] = 'Changed input CDELT1 and CDELT2'
        map_header['CDELT1'] = map_res
        map_header['CDELT2'] = map_res
        map_header['comment'] = 'Changed surface brightness units to ' + map_units
        map_header['comment'] = 'Repositioned the map on the sky'

        self.map_data = {"images":        map_image,
                         "shape":         map_image.shape,
                         "header":        map_header,
                         "center":        map_center,
                         "res":           map_res,
                         "inbright":      map_inbright,
                         "units":         map_units,
                         }

        if self.map_data["inbright"] is not None:
            self.map_data["images"] *= self.map_data["inbright"] / np.nanmax(self.map_data["images"])
            self.map_data["header"][""] = "Amplitude is rescaled."

        if self.map_data['units'] == 'Jy/pixel':
            for i in range(len(self.array.detectors)):
                self.map_data['images'][i] = self.map_data['images'][i]/utils.KbrightToJyPix(self.array.detectors[i][0], self.map_data['res'], self.map_data['res'])

    def run(self, **kwargs):

        self.sample_maps()
        self.temperature = self.map_samples[0]
        
    def sample_maps(self):
        
        map_res = np.radians(self.map_data["res"])
        n_maps, map_nx, map_ny = self.map_data["shape"]

        map_x = map_res * map_nx * np.linspace(-0.5, 0.5, map_nx)
        map_y = map_res * map_ny * np.linspace(-0.5, 0.5, map_ny)
        map_X, map_Y = np.meshgrid(map_x, map_y, indexing="ij")
        self.map_samples = np.zeros((self.X.shape))

        for i_map in range(n_maps):
            mask = (self.array.band == np.unique(self.array.band)[i_map])
            _map_samples = sp.interpolate.RegularGridInterpolator(
                (map_x, map_y), self.map_data["images"][i_map], bounds_error=False, fill_value=0)((self.X[mask], self.Y[mask]))
            self.map_samples[mask] = _map_samples
