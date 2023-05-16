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
from datetime import datetime

from astropy.io import fits

# how do we do the bands? this is a great question.
# because all practical telescope instrumentation assume a constant band

here, this_filename = os.path.split(__file__)

from . import simulations


class SkySimulation(simulations.BaseSimulation):
    """
    This simulates scanning over celestial sources.
    """
    def __init__(self, array, pointing, site, map_file, **kwargs):
        super().__init__(array, pointing, site)

        self.AZIM, self.ELEV = utils.from_xy(
            self.array.offset_x.values[:, None],
            self.array.offset_y.values[:, None],
            self.pointing.az,
            self.pointing.el,
        )

        self.RA, self.DEC = self.coordinator.transform(
            self.pointing.unix,
            self.AZIM,
            self.ELEV,
            in_frame="az_el",
            out_frame="ra_dec",
        )

        self.X, self.Y = utils.to_xy(self.RA, self.DEC, self.RA.mean(), self.DEC.mean())

        self.map_file = map_file

    def run(self, **kwargs):

        # get the CMB
        if self.add_cmb:
            self._get_CMBPS()

        # Get the astronomical signal
        self._get_skyconfig(**kwargs)
        self._get_sky()
        self._savesky()


    def _get_CMBPS(self):
        
        import camb

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        
        # correct mode would l=129600 for 5"
        pars.set_for_lmax(5000, lens_potential_accuracy=0) 

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="K")["total"][:, 0]

        self.CMB_PS = np.empty((len(np.unique(self.array.bands)), len(powers)))
        for i in range(len(np.unique(self.array.bands))):
            self.CMB_PS[i] = powers


    def _cmb_imager(self, bandnumber=0):

        import pymaster as nmt

        nx, ny = self.im[0].shape
        Lx = nx * np.deg2rad(self.sky_data["incell"])
        Ly = ny * np.deg2rad(self.sky_data["incell"])

        self.CMB_map = nmt.synfast_flat(
            nx,
            ny,
            Lx,
            Ly,
            np.array([self.CMB_PS[bandnumber]]),
            [0],
            beam=None,
            seed=self.pointing.seed,
        )[0]

        self.CMB_map += utils.Tcmb
        self.CMB_map *= utils.KcmbToKbright(np.unique(self.array.bands)[bandnumber])

    def _get_skyconfig(self, **kwargs):

        hudl = fits.open(self.file_name)
        self.im = hudl[0].data
        self.he = hudl[0].header

        if len(self.im.shape) == 4:
            self.im = self.im[0]
        elif len(self.im.shape) == 2:
            self.im = self.im.reshape(1,self.im.shape[0], self.im.shape[1])

        self.sky_data = {
            "inbright": kwargs.get("inbright", None),             # assuming something: Jy/pix?
            "incell":   kwargs.get("incell", self.he["CDELT1"]),  # assuming units in degree
            "units":    kwargs.get("units", 'KRJ'),               # Kelvin Rayleigh Jeans (KRJ) or Jy/pixel            
        }

        #updating header info
        self.he['HISTORY'] = 'History_WeOBSERVE 1'
        self.he[''] = 'Changed CDELT1 and CDELT2'
        self.he['CDELT1'] = self.sky_data['incell']
        self.he['CDELT2'] = self.sky_data['incell']
        self.he[''] = 'Changed units to ' + self.sky_data['units']

        if self.sky_data["inbright"] != None:
            self.im = self.im / np.nanmax(self.im) * self.sky_data["inbright"]
            self.he[''] = 'Amplitude is rescaled.'
        
    def _get_sky(
        self,
    ):
        
        map_res = np.radians(self.sky_data["incell"])
        map_nx, map_ny = self.im[0].shape
        map_x = map_res * map_nx * np.linspace(-0.5, 0.5, map_nx)
        map_y = map_res * map_ny * np.linspace(-0.5, 0.5, map_ny)
        map_X, map_Y = np.meshgrid(map_x, map_y, indexing="ij")

        map_x_bins = np.arange(map_X.min(), map_X.max(), 8 * map_res)
        map_y_bins = np.arange(map_Y.min(), map_Y.max(), 8 * map_res)

        self.truesky     = np.empty((len(np.unique(self.array.bands)),len(map_x_bins)-1,len(map_y_bins)-1))
        self.noisemap    = np.empty((len(np.unique(self.array.bands)),len(map_x_bins)-1,len(map_y_bins)-1))
        self.filteredmap = np.empty((len(np.unique(self.array.bands)),len(map_x_bins)-1,len(map_y_bins)-1))
        self.mockobs     = np.empty((len(np.unique(self.array.bands)),len(map_x_bins)-1,len(map_y_bins)-1))

        # should mask the correct detectors...
        for iub, band in enumerate(self.array.ubands):
            mask = self.array.band == band
            self._make_sky(self.X, self.Y, map_y, map_X, map_Y, map_x_bins, map_y_bins, iub, mask)

    def _make_sky(
        self,  
        lam_x,
        lam_y,   
        map_x,
        map_y, 
        map_X,
        map_Y,
        x_bins,
        y_bins,
        i,
        mask
    ):

        if self.im.shape[0] == 1:
            idx = 0
        else:
            idx = i

        if self.sky_data['units'] == 'Jy/pixel':
            self.im[idx] = self.im[idx]/utils.KbrightToJyPix(np.unique(self.array.bands)[i], self.sky_data['incell'], self.sky_data['incell'])

        self.map_data = sp.interpolate.RegularGridInterpolator(
            (map_x, map_y), self.im[idx], bounds_error=False, fill_value=0
        )((lam_x, lam_y))

        if self.add_cmb:
            self._cmb_imager(i)
            cmb_data = sp.interpolate.RegularGridInterpolator(
                        (map_x, map_y), self.CMB_map, bounds_error=False, fill_value=0
                        )((lam_x, lam_y))
            self.noise    = self.lam.temperature + cmb_data
            self.combined = self.map_data + self.lam.temperature + cmb_data
        else:
            self.noise    = self.lam.temperature
            self.combined = self.map_data + self.lam.temperature

        true_map = sp.stats.binned_statistic_2d(
            map_X.ravel(),
            map_Y.ravel(),
            self.im[idx].ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        filtered_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            self.map_data.ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        total_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            self.combined[i].ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        noise_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            self.noise[i].ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        self.truesky[i]     = true_map
        self.noisemap[i]    = noise_map
        self.filteredmap[i] = filtered_map
        self.mockobs[i]     = total_map

        if self.sky_data['units'] == 'Jy/pixel':
            self.truesky[i]     *= utils.KbrightToJyPix(np.unique(self.array.bands)[i], self.sky_data['incell'], self.sky_data['incell'])
            self.noisemap[i]    *= utils.KbrightToJyPix(np.unique(self.array.bands)[i], self.sky_data['incell'], self.sky_data['incell'])
            self.filteredmap[i] *= utils.KbrightToJyPix(np.unique(self.array.bands)[i], self.sky_data['incell'], self.sky_data['incell'])
            self.mockobs[i]     *= utils.KbrightToJyPix(np.unique(self.array.bands)[i], self.sky_data['incell'], self.sky_data['incell'])
