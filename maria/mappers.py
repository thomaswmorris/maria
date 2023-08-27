import numpy as np
import scipy as sp
import pandas as pd
import os
from tqdm import tqdm
import warnings
from importlib import resources
import time as ttime
from . import utils
import weathergen
from os import path
import json
from datetime import datetime
import healpy as hp

import matplotlib.pyplot as plt
from .coordinator import Coordinator
from astropy.io import fits

here, this_filename = os.path.split(__file__)

MAPPER_CONFIGS = utils.read_yaml(f"{here}/configs/mappers.yml")
MAPPERS = list((MAPPER_CONFIGS.keys()))




class InvalidMapperError(Exception):
    def __init__(self, invalid_mapper):
        print(f"The mapper \'{invalid_mapper}\' is not in the database of default mappers."
              f"Default mappers are: {MAPPERS}")

class BaseMapper:
    """
    The base class for modeling atmospheric fluctuations.

    A model needs to have the functionality to generate spectra for any pointing data we supply it with.
    """

    def __init__(self, **kwargs):

        self.tods = []
        self.map_res    = kwargs.get("map_res", np.radians(1/60))
        self.map_width  = kwargs.get("map_width", np.radians(5))
        self.map_height = kwargs.get("map_height", np.radians(5))

        # self.header = fits.header.Header()

    @property
    def maps(self):
        return {key:self.map_sums[key]/np.where(self.map_cnts[key], self.map_cnts[key], np.nan) for key in self.map_sums.keys()}

    def smoothed_maps(self, smoothing=1):

        smoothed_maps = {}

        for key in self.map_sums.keys():

            SUMS = sp.ndimage.gaussian_filter(self.map_sums[key], sigma=(smoothing, smoothing))
            CNTS = sp.ndimage.gaussian_filter(self.map_cnts[key], sigma=(smoothing, smoothing))

            smoothed_maps[key] = SUMS / CNTS

        return smoothed_maps

    def expand_tod(self, tod):

        coordinator = Coordinator(lat=tod.meta['latitude'], lon=tod.meta['longitude'])

        tod.AZ, tod.EL = utils.xy_to_lonlat(
            tod.dets.sky_x.values[:, None],
            tod.dets.sky_y.values[:, None],
            tod.az,
            tod.el,
        )

        tod.LON, tod.LAT = coordinator.transform(
            tod.time,
            tod.AZ,
            tod.EL,
            in_frame="az_el",
            out_frame="ra_dec",
        )

        return tod

    def add_tods(self, tods):

        for tod in np.atleast_1d(tods):
            self.tods.append(self.expand_tod(tod))

    @property
    def get_map_center_lonlat(self):

        for tod in self.tods:
            mean_unit_vec  = hp.ang2vec(np.pi/2 - tod.LAT.ravel(), tod.LON.ravel()).mean(axis=0)
            mean_unit_vec /= np.sqrt(np.sum(np.square(mean_unit_vec)))
            mean_unit_colat, mean_unit_lon = np.r_[hp.vec2ang(mean_unit_vec)]

        return mean_unit_lon, np.pi/2 - mean_unit_colat
    
    def save_maps(self, filepath):

        self.header = self.tods[0].header
        self.header['comment'] = 'Made Synthetic observations via maria code'
        self.header['comment'] = 'Overwrote resolution and size of the output map'
        self.header['CDELT1']  = np.rad2deg(self.map_res)
        self.header['CDELT2']  = np.rad2deg(self.map_res)
        self.header['CRPIX1']  = self.maps[list(self.maps.keys())[0]].shape[0]/2
        self.header['CRPIX2']  = self.maps[list(self.maps.keys())[0]].shape[1]/2

        self.header['CRVAL1'] = np.rad2deg(self.tods[0].cntr[0])
        self.header['CRVAL2'] = np.rad2deg(self.tods[0].cntr[1])

        self.header['CTYPE1']  = 'RA---SIN'
        self.header['CUNIT1']  = 'deg     ' 
        self.header['CTYPE2']  = 'DEC--SIN' 
        self.header['CUNIT2']  = 'deg     '
        self.header['comment'] = 'Overwrote pointing location of the output map'

        self.header['comment'] = 'Overwrote spectral position of the output map'
        self.header['CTYPE3']  = 'FREQ    '
        self.header['CUNIT3']  = 'Hz      '
        self.header['CRPIX3']  = 1.000000000000E+00
        
        self.header['BTYPE']   = 'Intensity'
        if self.tods[0].unit == 'Jy/pixel': 
            self.header['BUNIT']   = 'Jy/pixel '   
        else: 
            self.header['BUNIT']   = 'Kelvin RJ'   

        for i, key in enumerate(self.maps.keys()):
            
            # what is this? --> Frequency information in the header
            self.header['CRVAL3'] = self.nom_freqs[key]
            # self.header['CDELT3']  = self.array.dets[i][1]

            save_map = self.maps[list(self.maps.keys())[i]] 

            if self.tods[0].unit == 'Jy/pixel': 
                save_map *= utils.KbrightToJyPix(self.header['CRVAL3'], 
                                                 np.deg2rad(self.header['CDELT1']), 
                                                 np.deg2rad(self.header['CDELT2'])
                                                )
                
            fits.writeto( filename = filepath.split('.fits')[0] + '_'+list(self.maps.keys())[i]+'.fits', 
                            data = save_map, 
                            header = self.header,
                            overwrite = True 
                        )




class BinMapper(BaseMapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nmtr = kwargs.get("n_modes_to_remove", 0)
        self.x_bins = np.arange(-0.5*self.map_width, 0.5*self.map_width, self.map_res)
        self.y_bins = np.arange(-0.5*self.map_height, 0.5*self.map_height, self.map_res)
        self.n_x, self.n_y = len(self.x_bins) - 1, len(self.y_bins) - 1

    def run(self):

        self.ubands = sorted([band for tod in self.tods for band in np.unique(tod.dets.band)])
        self.nom_freqs = {}
        self.map_sums = {band: np.zeros((self.n_x, self.n_y)) for band in self.ubands}
        self.map_cnts = {band: np.zeros((self.n_x, self.n_y)) for band in self.ubands}

        for band in np.unique(self.ubands):
            for tod in self.tods:

                band_mask = tod.dets.band == band

                LON, LAT = tod.LON[band_mask], tod.LAT[band_mask]
                u, s, v = np.linalg.svd(sp.signal.detrend(tod.data), full_matrices=False)
                DATA = utils.mprod(u[:, self._nmtr:], np.diag(s[self._nmtr:]), v[self._nmtr:])

                #pointing_in_rel_map_units_X, pointing_in_rel_map_units_Y = utils.lonlat_to_xy(self.RA, self.LAT, self.map.center[0], self.map.center[1])
                
                X, Y = utils.lonlat_to_xy(LON, LAT, *self.get_map_center_lonlat)

                map_sum = sp.stats.binned_statistic_2d(X.ravel(), 
                                                       Y.ravel(),
                                                       DATA.ravel(),
                                                       bins=(self.x_bins, self.y_bins),
                                                       statistic='sum')[0]

                map_cnt = sp.stats.binned_statistic_2d(X[band_mask].ravel(), 
                                                       Y[band_mask].ravel(),
                                                       DATA[band_mask].ravel(),
                                                       bins=(self.x_bins, self.y_bins),
                                                       statistic='count')[0]
                self.map_sums[band] += map_sum
                self.map_cnts[band] += map_cnt

                self.nom_freqs[band] = tod.dets.band_center.mean()
            
