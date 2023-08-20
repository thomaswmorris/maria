import numpy as np
import scipy as sp
import pandas as pd
import astropy as ap
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

from dataclasses import dataclass, field

@dataclass
class Map():
    
    data: np.array
    freqs: np.array
    res: float
    inbright: float
    center: tuple
    header: ap.io.fits.header.Header = None
    frame: str = "ra_dec"
    units: str = "K"
    
    @property
    def n_freqs(self):
        return len(self.freqs)

    @property
    def shape(self):
        return self.data.shape[-2:]
    
    @property
    def n_x(self):
        return self.shape[0]
    
    @property
    def n_y(self):
        return self.shape[1]
    
    @property
    def rel_x_side(self):
        x = self.res * np.arange(self.n_x)
        return x - x.mean()
        
    @property
    def rel_y_side(self):
        y = self.res * np.arange(self.n_y)
        return y - y.mean()
    
    @property
    def x_side(self):
        return self.rel_x_side + self.center[0]
    
    @property
    def y_side(self):
        return self.rel_y_side + self.center[1]
    
    @property
    def X_Y(self):
        return np.meshgrid(self.x_side, self.y_side)

    @property
    def rel_X_Y(self):
        return np.meshgrid(self.rel_x_side, self.rel_y_side)
    
    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=128)
        map_extent = np.degrees([self.rel_x_side.min(), self.rel_x_side.max(), self.rel_y_side.min(), self.rel_y_side.max()])
        map_im = ax.imshow(self.data[0], extent=map_extent)
        if self.frame == "ra_dec":
            ax.set_xlabel("RA (deg.)")
            ax.set_ylabel("dec. (deg.)")
        clb = fig.colorbar(mappable=map_im, shrink=0.8, aspect=32)
        clb.set_label(self.units)


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

        AZIM, ELEV = utils.x_y_to_phi_theta(
            self.array.sky_x[:, None],
            self.array.sky_y[:, None],
            self.pointing.az,
            self.pointing.el,
        )

        self.RA, self.DEC = self.coordinator.transform(
            self.pointing.unix,
            AZIM, ELEV,
            in_frame="az_el",
            out_frame="ra_dec",
        )

        self.X, self.Y = utils.phi_theta_to_x_y(self.RA, self.DEC, self.RA.mean(), self.DEC.mean())


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

        self.map_file = map_file
        hudl = ap.io.fits.open(map_file)

        map_res_radians = np.radians(kwargs.get("map_res", 1/1000))
        map_center_radians = np.radians(kwargs.get("map_center", (10.5, 4)))

        self.map = Map(data=hudl[0].data[None],
                       freqs=np.atleast_1d(kwargs.get("map_freqs", 90e9)),
                       res=map_res_radians,
                       center=map_center_radians,
                       frame=kwargs.get("map_frame", "ra_dec"),
                       inbright=kwargs.get("inbright", None),
                       header=hudl[0].header,
                       units="K")

        # # Prep mapmaker
        # # -------------
        # self.map_file = map_file
        # hudl          = ap.io.fits.open(self.map_file)
        # map_image     = hudl[0].data

        # if map_image.ndim == 2:
        #     map_image = map_image[None]

        # *_, map_nx, map_ny = map_image.shape

        # #map_image = map_image.reshape(-1, map_nx, map_ny)

        # if len(self.array.dets) != map_image.shape[0] | map_image.shape[0] != 1:
        #     raise InvalidNBandsError(len(self.array.dets))

        # map is a degree wide by default
        # DEFAULT_MAP_CENTER = (self.RA.mean(), self.DEC.mean())
        # DEFAULT_MAP_RES = 1 / np.maximum(map_nx, map_ny)

        # map_center   = kwargs.get("map_center",     DEFAULT_MAP_CENTER)
        # map_res      = kwargs.get("map_res",        DEFAULT_MAP_RES)
        # map_units    = kwargs.get("map_units",     'KRJ')
        # map_inbright = kwargs.get("map_inbright",   None)
        
        self.map.header['HISTORY'] = 'History_input_adjustments'
        self.map.header['comment'] = 'Changed input CDELT1 and CDELT2'
        self.map.header['CDELT1']  = self.map.res
        self.map.header['CDELT2']  = self.map.res
        self.map.header['comment'] = 'Changed surface brightness units to ' + self.map.units
        self.map.header['comment'] = 'Repositioned the map on the sky'

        # self.map_data = {"images":        map_image,
        #                  "shape":         map_image.shape,
        #                  "header":        self.map.header,
        #                  "center":        map_center,
        #                  "res":           map_res,
        #                  "inbright":      map_inbright,
        #                  "units":         map_units,
        #                  }

        if self.map.inbright is not None:
            self.map.data *= self.map.inbright / np.nanmax(self.map.data)
            self.map.header[""] = "Amplitude is rescaled."

        if self.map.units == 'Jy/pixel':
            for i, nu in enumerate(self.map.freqs):
                self.map.data[i] = self.map.data[i] / utils.KbrightToJyPix(nu, self.map.res, self.map.res)

    def run(self, **kwargs):
        self.sample_maps()
        self.temperature = self.map_samples[0]
        
    def sample_maps(self):

        pointing_in_rel_map_units_X, pointing_in_rel_map_units_Y = utils.phi_theta_to_x_y(self.RA, self.DEC, self.map.center[0], self.map.center[1])
        
        self.map_samples = np.zeros((self.RA.shape))

        for i, nu in enumerate(self.map.freqs):

            det_freq_response = self.array.passband(nu=np.array([nu]))[:,0]

            det_mask = det_freq_response > 1e-3

            samples = sp.interpolate.RegularGridInterpolator((self.map.rel_x_side, self.map.rel_x_side), self.map.data[i], bounds_error=False, fill_value=0)((pointing_in_rel_map_units_X[det_mask], pointing_in_rel_map_units_Y[det_mask]))
            
            self.map_samples[det_mask] = samples
