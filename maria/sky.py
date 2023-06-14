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

here, this_filename = os.path.split(__file__)

from . import base

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

        self.map_file = map_file

        hudl = fits.open(self.map_file)
        map_image = hudl[0].data
        *_, map_nx, map_ny = map_image.shape
        map_image = map_image.reshape(-1, map_nx, map_ny)

        DEFAULT_MAP_CENTER = (self.RA.mean(), self.DEC.mean())
        DEFAULT_MAP_RES = 1 / np.maximum(map_nx, map_ny) # map is a degree wide by default

        map_center = kwargs.get("map_center", DEFAULT_MAP_CENTER)
        map_res    = kwargs.get("map_res", DEFAULT_MAP_RES)
        map_units  = kwargs.get("units", 'KRJ')
        inbright = kwargs.get("inbright", None)
        
        map_header = hudl[0].header
        map_header['HISTORY'] = 'History_WeOBSERVE 1'
        map_header[''] = 'Changed CDELT1 and CDELT2'
        map_header['CDELT1'] = map_res
        map_header['CDELT2'] = map_res
        map_header[''] = 'Changed units to ' + map_units

        self.map_data = {"images": map_image,
                         "shape": map_image.shape,
                         "header": map_header,
                         "center": map_center,
                         "res": map_res,
                         "inbright": inbright,
                         "units": map_units,
                         }

        if self.map_data["inbright"] is not None:
            self.map_data["images"] *= self.map_data["inbright"] / np.nanmax(self.map_data["images"])
            self.map_data["header"][""] = "Amplitude is rescaled."

    def run(self, **kwargs):

        self.sample_maps()
        self.temperature = self.map_samples[0]

        
    def sample_maps(self):
        
        map_res = np.radians(self.map_data["res"])
        n_maps, map_nx, map_ny = self.map_data["shape"]

        map_x = map_res * map_nx * np.linspace(-0.5, 0.5, map_nx)
        map_y = map_res * map_ny * np.linspace(-0.5, 0.5, map_ny)
        map_X, map_Y = np.meshgrid(map_x, map_y, indexing="ij")

        # map_x_bins = np.arange(map_X.min(), map_X.max(), 8 * map_res)
        # map_y_bins = np.arange(map_Y.min(), map_Y.max(), 8 * map_res)

        # self.truesky     = np.empty((len(np.unique(self.array.band)),len(map_x_bins)-1,len(map_y_bins)-1))
        # self.noisemap    = np.empty((len(np.unique(self.array.band)),len(map_x_bins)-1,len(map_y_bins)-1))
        # self.filteredmap = np.empty((len(np.unique(self.array.band)),len(map_x_bins)-1,len(map_y_bins)-1))
        # self.mockobs     = np.empty((len(np.unique(self.array.band)),len(map_x_bins)-1,len(map_y_bins)-1))

        # should mask the correct detectors...

        self.map_samples = np.zeros((0, *self.X.shape))

        for i_map in range(n_maps):

            _map_samples = sp.interpolate.RegularGridInterpolator(
                (map_x, map_y), self.map_data["images"][i_map], bounds_error=False, fill_value=0)((self.X, self.Y))
            self.map_samples = np.r_[self.map_samples, _map_samples[None]]

        

        # for iub, band in enumerate(self.array.ubands):
        #     mask = self.array.band == band
        #     self._make_sky(self.X, self.Y, map_y, map_X, map_Y, map_x_bins, map_y_bins, iub, mask)

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
    ):

        if self.map_data.shape[0] == 1:
            idx = 0
        else:
            idx = i

        if self.sky_data['units'] == 'Jy/pixel':
            self.map_data[idx] = self.map_data[idx]/utils.KbrightToJyPix(np.unique(self.array.band_center)[i], self.sky_data['incell'], self.sky_data['incell'])

        self.map_data = sp.interpolate.RegularGridInterpolator(
            (map_x, map_y), self.map_data[idx], bounds_error=False, fill_value=0
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
            self.map_data[idx].ravel(),
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
            self.truesky[i]     *= utils.KbrightToJyPix(np.unique(self.array.band)[i], self.sky_data['incell'], self.sky_data['incell'])
            self.noisemap[i]    *= utils.KbrightToJyPix(np.unique(self.array.band)[i], self.sky_data['incell'], self.sky_data['incell'])
            self.filteredmap[i] *= utils.KbrightToJyPix(np.unique(self.array.band)[i], self.sky_data['incell'], self.sky_data['incell'])
            self.mockobs[i]     *= utils.KbrightToJyPix(np.unique(self.array.band)[i], self.sky_data['incell'], self.sky_data['incell'])


    def _analysis(self):

        if not os.path.exists(self.file_save + "/analyzes"):
            os.mkdir(self.file_save + "/analyzes")

        # visualize scanning patern
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=256, tight_layout=True)
        axes[0].plot(np.degrees(self.lam.pointing.az), np.degrees(self.lam.pointing.el), lw=5e-1)
        axes[0].set_xlabel("az (deg)"), axes[0].set_ylabel("el (deg)")
        axes[1].plot(np.degrees(self.lam.pointing.ra), np.degrees(self.lam.pointing.dec), lw=5e-1)
        axes[1].set_xlabel("ra (deg)"), axes[1].set_ylabel("dec (deg)")
        plt.savefig(self.file_save + "/analyzes/scanpattern_" + self.map_file.replace(".fits", "").split("/")[-1] + ".png")
        plt.close()

        # visualize powerspectrum
        f, ps = sp.signal.periodogram(self.lam.temperature[0], fs=self.lam.pointing.sample_rate, window="tukey")
        plt.figure()
        plt.plot(f[1:], ps.mean(axis=0)[1:], label="atmosphere")
        plt.plot(f[1:], f[1:] ** (-8 / 3), label="y = f^-(8/3)")
        plt.loglog()
        plt.xlabel("l")
        plt.ylabel("PS")
        plt.legend()
        plt.savefig(self.file_save + "/analyzes/Noise_ps_" + self.map_file.replace(".fits", "").split("/")[-1] + ".png")
        plt.close()




        # visualize fits files
        fig, (true_ax, signal_ax, noise_ax, total_ax) = plt.subplots(
            1, 4, figsize=(9, 3), sharex=True, sharey=True, constrained_layout=True
        )

        total_plt = true_ax.imshow(self.truesky[0])
        true_ax.set_title("True map")
        fig.colorbar(total_plt, ax=true_ax, location="bottom", shrink=0.8)

        true_plt = signal_ax.imshow(self.filteredmap[0])
        signal_ax.set_title("Filtered map")
        fig.colorbar(true_plt, ax=signal_ax, location="bottom", shrink=0.8)

        signal_plt = noise_ax.imshow(self.noisemap[0])
        noise_ax.set_title("Noise map")
        fig.colorbar(signal_plt, ax=noise_ax, location="bottom", shrink=0.8)

        total_plt = total_ax.imshow(self.mockobs[0])
        total_ax.set_title("Synthetic Observation")
        fig.colorbar(total_plt, ax=total_ax, location="bottom", shrink=0.8)

        plt.savefig(self.file_save + "/analyzes/maps_" + self.map_file.replace(".fits", "").split("/")[-1] + ".png")
        plt.close()



class CMBSimulation(base.BaseSimulation):
    """
    This simulates scanning over celestial sources.
    """
    def __init__(self, array, pointing, site, map_file, add_cmb=False, **kwargs):
        super().__init__(array, pointing, site)

        pass

    

    def _get_CMBPS(self):
        
        import camb

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        
        # correct mode would l=129600 for 5"
        pars.set_for_lmax(5000, lens_potential_accuracy=0) 

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="K")["total"][:, 0]

        self.CMB_PS = np.empty((len(np.unique(self.array.band)), len(powers)))
        for i in range(len(np.unique(self.array.band))):
            self.CMB_PS[i] = powers


    def _cmb_imager(self, bandnumber=0):

        import pymaster as nmt

        nx, ny = self.map_data[0].shape
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
        self.CMB_map *= utils.KcmbToKbright(np.unique(self.array.band_center)[bandnumber])


