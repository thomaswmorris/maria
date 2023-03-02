from . import utils

import numpy as np
from numpy import linalg as la

import scipy as sp

import weathergen
sites = weathergen.sites

class Array():
    def __init__(self, config, verbose=False):

        self.config = config

        for k, v in config.items(): setattr(self, k, v)

        self.field_of_view = self.config['field_of_view']
        self.primary_size  = self.config['primary_size']
       
        self.bands, self.bandwidths, self.n_det = np.empty(0), np.empty(0), 0
        for nu_0, nu_w, n in self.config['bands']:
            self.bands, self.bandwidths = np.r_[self.bands, np.repeat(nu_0, n)], np.r_[self.bandwidths, np.repeat(nu_w, n)]
            self.n_det += n

        self.offsets  = utils.make_array(self.config['geometry'], self.field_of_view, self.n_det)
        self.offsets *= np.pi / 180 # put these in radians
        
        # compute detector offsets
        self.hull = sp.spatial.ConvexHull(self.offsets)

        # scramble up the locations of the bands 
        if self.config['band_grouping'] == 'random':
            random_index = np.random.choice(np.arange(self.n_det),self.n_det,replace=False)
            self.offsets = self.offsets[random_index]

        self.offset_x, self.offset_y = self.offsets.T
        self.r, self.p = np.sqrt(np.square(self.offsets).sum(axis=1)), np.arctan2(*self.offsets.T)

        self.ubands = np.unique(self.bands)
        self.nu = np.arange(0, 1e12, 1e9)
        
        self.passbands  = np.c_[[utils.get_passband(self.nu, nu_0, nu_w, order=16) for nu_0, nu_w in zip(self.bands, self.bandwidths)]]

        nu_mask = (self.passbands > 1e-4).any(axis=0)
        self.nu, self.passbands = self.nu[nu_mask], self.passbands[:, nu_mask]

        self.passbands /= self.passbands.sum(axis=1)[:,None]

        # compute beams
        self.optical_model = 'diff_lim'
        if self.optical_model == 'diff_lim':

            self.get_beam_waist   = lambda z, w_0, f : w_0 * np.sqrt(1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0)))
            self.get_beam_profile = lambda r, r_fwhm : np.exp(np.log(0.5)*np.abs(r/r_fwhm)**8)
            self.beam_func = self.get_beam_profile

        # position detectors:
        self.site = self.config['site']
        self.latitude, self.longitude, self.altitude = sites.loc[sites.index == self.site, ['latitude', 'longitude', 'altitude']].values[0]

        self.coordinator  = utils.coordinator(lat=self.latitude, lon=self.longitude)

    def make_filter(self, waist, res, func, width_per_waist=1.2):
    
        filter_width = width_per_waist * waist
        n_filter = 2 * int(np.ceil(0.5 * filter_width / res)) + 1

        filter_side = 0.5 * np.linspace(-filter_width, filter_width, n_filter)

        FILTER_X, FILTER_Y = np.meshgrid(filter_side, filter_side, indexing='ij')
        FILTER_R = np.sqrt(np.square(FILTER_X) + np.square(FILTER_Y))

        FILTER  = func(FILTER_R, 0.5 * waist)
        FILTER /= FILTER.sum()
        
        return FILTER
    
    def separate_filter(self, F, tol=1e-2):
        
        u, s, v = la.svd(F); eff_filter = 0
        for m, (_u, _s, _v) in enumerate(zip(u.T, s, v)):

            eff_filter += _s * np.outer(_u, _v)
            if np.abs(F - eff_filter).sum() < tol: break

        return u.T[:m+1], s[:m+1], v[:m+1]

    def separably_filter(self, M, F, tol=1e-2):

        u, s, v = self.separate_filter(F, tol=tol)

        filt_M = 0
        for _u, _s, _v in zip(u, s, v):

            filt_M += _s * sp.ndimage.convolve1d(sp.ndimage.convolve1d(M.astype(float), _u, axis=0), _v, axis=1)

        return filt_M