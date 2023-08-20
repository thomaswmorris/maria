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

import matplotlib.pyplot as plt

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
        self.resolution = kwargs.get("resolution", np.radians(1/60))

    @property
    def maps(self):
        return {key:self.map_sums[key]/np.where(self.map_cnts[key], self.map_cnts[key], np.nan) for key in self.map_sums.keys()}

    def expand_tod(self, tod):

        coordinator = utils.Coordinator(lat=tod.meta['latitude'], 
                                        lon=tod.meta['longitude'])

        tod.AZ, tod.EL = utils.x_y_to_phi_theta(
            tod.dets.sky_x.values[:, None],
            tod.dets.sky_y.values[:, None],
            tod.az,
            tod.el,
        )

        tod.RA, tod.DEC = coordinator.transform(
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


class BinMapper(BaseMapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nmtr = kwargs.get("n_modes_to_remove", 0)

    def run(self):

        self.ubands = sorted([band for tod in self.tods for band in np.unique(tod.dets.band)])

        min_ra = np.min([tod.RA.min() for tod in self.tods])
        max_ra = np.max([tod.RA.max() for tod in self.tods])

        min_dec = np.min([tod.DEC.min() for tod in self.tods])
        max_dec = np.max([tod.DEC.max() for tod in self.tods])

        self.ra_bins = np.arange(min_ra, max_ra, self.resolution)
        self.dec_bins = np.arange(min_dec, max_dec, self.resolution)

        self.ra_side = 0.5 * (self.ra_bins[1:] + self.ra_bins[:-1])
        self.dec_side = 0.5 * (self.dec_bins[1:] + self.dec_bins[:-1])

        self.n_ra, self.n_dec = len(self.ra_bins) - 1, len(self.dec_bins) - 1

        self.map_sums = {band: np.zeros((self.n_ra, self.n_dec)) for band in self.ubands}
        self.map_cnts = {band: np.zeros((self.n_ra, self.n_dec)) for band in self.ubands}

        for band in np.unique(self.ubands):
            for tod in self.tods:

                band_mask = tod.dets.band == band

                RA, DEC = tod.RA[band_mask], tod.DEC[band_mask]
                u, s, v = np.linalg.svd(sp.signal.detrend(tod.data), full_matrices=False)
                DATA = utils.mprod(u[:, self._nmtr:], np.diag(s[self._nmtr:]), v[self._nmtr:])

                
                map_sum = sp.stats.binned_statistic_2d(RA.ravel(), 
                                                       DEC.ravel(),
                                                       DATA.ravel(),
                                                       bins=(self.ra_bins, self.dec_bins),
                                                       statistic='sum')[0]

                map_cnt = sp.stats.binned_statistic_2d(RA[band_mask].ravel(), 
                                                       DEC[band_mask].ravel(),
                                                       DATA[band_mask].ravel(),
                                                       bins=(self.ra_bins, self.dec_bins),
                                                       statistic='count')[0]
                self.map_sums[band] += map_sum
                self.map_cnts[band] += map_cnt
