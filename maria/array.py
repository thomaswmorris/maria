import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt

import os

from . import utils

from collections.abc import Mapping


here, this_filename = os.path.split(__file__)

# -- Specific packages --
ARRAY_CONFIGS = utils.read_yaml(f'{here}/configs/arrays.yml')

ARRAYS = list((ARRAY_CONFIGS.keys()))


class UnsupportedArrayError(Exception):
    def __init__(self, invalid_array):
        super().__init__(f"The array \'{invalid_array}\' is not in the database of default arrays. "
        f"Default arrays are:\n\n{sorted(list(ARRAY_CONFIGS.keys()))}")


def get_array_config(array_name, **kwargs):
    if not array_name in ARRAY_CONFIGS.keys():
        raise UnsupportedArrayError(array_name)
    ARRAY_CONFIG = ARRAY_CONFIGS[array_name].copy()
    for k, v in kwargs.items():
        ARRAY_CONFIG[k] = v
    return ARRAY_CONFIG


def get_array(array_name, **kwargs):
    return Array(**get_array_config(array_name, **kwargs))


def get_array_from_fits(array_name, **kwargs):
    return Array(**get_array_config(array_name, **kwargs))


class Array:
    def __init__(self, **kwargs):

        DEFAULT_ARRAY_CONFIG = {
        "detectors": [
            [150e9, 10e9, 100], 
        ],
        "geometry": "hex", 
        "field_of_view": 1.3, 
        "primary_size": 50,   
        "band_grouping": "randomized",   
        "az_bounds": [0, 360],  
        "el_bounds": [20, 90],
        "max_az_vel": 3,
        "max_el_vel": 2,
        "max_az_acc": 1,
        "max_el_acc": 0.25
        }
        for k, v in kwargs.items():
            setattr(self, k, v)

        detectors = kwargs.get("detectors", "")
        
        if type(detectors) == pd.DataFrame:
            self.dets = detectors

        elif isinstance(detectors, Mapping):    

            self.dets = pd.DataFrame()

            for band, (band_center, band_width, n) in detectors.items():
            #for band in self.detectors.keys():

                band_dets = pd.DataFrame(index=np.arange(n))
                band_dets.loc[:, "det_uid"] = np.arange(n)
                band_dets.loc[:, "band"] = band
                band_dets.loc[:, "band_center"] = band_center
                band_dets.loc[:, "band_width"] = band_width

                self.dets = pd.concat([self.dets, band_dets])

            self.offset  = utils.make_array(self.geometry, self.field_of_view, self.n_dets)
            self.offset *= np.pi / 180  # put these in radians

            # scramble up the locations of the bands
            if self.band_grouping == "randomized":
                random_index = np.random.choice(np.arange(self.n_dets), self.n_dets, replace=False)
                self.offset = self.offset[random_index]

            self.sky_x, self.sky_y = self.offset.T
            self.r, self.p = np.sqrt(np.square(self.offset).sum(axis=1)), np.arctan2(*self.offset.T)

        else:
            raise ValueError("Supplied arg 'detectors' must be either a mapping or a dataframe!")
        
        #self.band = np.array([f"f{int(nu/10**(3*int(np.log10(nu)/3))):03}" for nu in self.band_center])



        

        # compute detector offset
        self.hull = sp.spatial.ConvexHull(self.offset)

        # compute beams
        self.optical_model = "diff_lim"
        if self.optical_model == "diff_lim":

            self.get_beam_waist = lambda z, w_0, f: w_0 * np.sqrt(
                1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0))
            )
            self.get_beam_profile = lambda r, r_fwhm: np.exp(np.log(0.5) * np.abs(r / r_fwhm) ** 8)
            self.beam_func = self.get_beam_profile

        self.dets.loc[:, "sky_x"] = self.sky_x
        self.dets.loc[:, "sky_y"] = self.sky_y

    @property
    def band_min(self):
        return (self.dets.band_center - 0.5 * self.dets.band_width).values

    @property
    def band_max(self):
        return (self.dets.band_center + 0.5 * self.dets.band_width).values

    def passband(self, nu):
        """
        Returns a (n_dets, len(nu))
        """
        return ((nu[None] > self.band_min[:, None]) & (nu[None] < self.band_max[:, None])).astype(float)

    @property
    def n_dets(self):
        return len(self.dets)

    @property
    def ubands(self):
        return list(np.unique(self.dets.band))

    def __repr__(self):

        return (f"Array Object"

                f"\nubands: {self.ubands})"
                f"\nn_dets: {self.n_dets})")
         
    def plot_dets(self):

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=160)

        for uband in self.ubands:

            band_mask = self.dets.band == uband

            ax.scatter(np.degrees(self.dets.sky_x[band_mask]), 
                       np.degrees(self.dets.sky_y[band_mask]),
                       label=uband, lw=5e-1)

        ax.set_xlabel(r'$\theta_x$ offset (deg.)')
        ax.set_ylabel(r'$\theta_y$ offset (deg.)')
        ax.legend()

    def make_filter(self, waist, res, func, width_per_waist=1.2):

        filter_width = width_per_waist * waist
        n_filter = 2 * int(np.ceil(0.5 * filter_width / res)) + 1

        filter_side = 0.5 * np.linspace(-filter_width, filter_width, n_filter)

        FILTER_X, FILTER_Y = np.meshgrid(filter_side, filter_side, indexing="ij")
        FILTER_R = np.sqrt(np.square(FILTER_X) + np.square(FILTER_Y))

        FILTER = func(FILTER_R, 0.5 * waist)
        FILTER /= FILTER.sum()

        return FILTER

    def separate_filter(self, F, tol=1e-2):

        u, s, v = np.linalg.svd(F)
        eff_filter = 0
        for m, (_u, _s, _v) in enumerate(zip(u.T, s, v)):

            eff_filter += _s * np.outer(_u, _v)
            if np.abs(F - eff_filter).sum() < tol:
                break

        return u.T[: m + 1], s[: m + 1], v[: m + 1]

    def separably_filter(self, M, F, tol=1e-2):

        u, s, v = self.separate_filter(F, tol=tol)

        filt_M = 0
        for _u, _s, _v in zip(u, s, v):

            filt_M += _s * sp.ndimage.convolve1d(sp.ndimage.convolve1d(M.astype(float), _u, axis=0), _v, axis=1)

        return filt_M

