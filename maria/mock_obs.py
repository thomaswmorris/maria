# -- General packages --
import os
import numpy as np
import scipy.signal
import scipy as sp

from matplotlib import pyplot as plt
from astropy.io import fits

from . import get_array, get_site, get_pointing
from . import models, utils, mappers

class WeObserve:
    def __init__(self, 
                 project, 
                 skymodel, 
                 array_name='MUSTANG-2', 
                 pointing_name='DAISY_2deg',
                 site_name='GBT',
                 mappingtype='Standard Binning',
                 verbose=True, 
                 cmb = False,
                 **kwargs):

        self.verbose     = verbose
        self.file_name   = skymodel
        self.file_save   = project
        self.add_cmb     = cmb
        self.mappingtype = mappingtype

        self.array    = get_array(array_name, **kwargs)
        self.pointing = get_pointing(pointing_name, **kwargs)
        self.site     = get_site(site_name, **kwargs)

        # get the atmosphere
        self._run_atmos()

        # Get the astronomical signal
        self._get_skyconfig(**kwargs)
        self._get_skytod()
        
        # get the CMB
        if self.add_cmb:
            self._get_CMBPS()
            self._cmb_imager()

        print(self.lam.temperature.shape)

        # Make the maps
        self.mapmaker = mappers.get_mapper(self.mappingtype, 
                                           self.sky_data, 
                                           self.array, 
                                           self.lam, 
                                           self.im, 
                                           self.he, 
                                           self.file_name)

        self.mapmaker.run(self.file_save +'/filtered_' )
        self.mapmaker.add_tod(self.lam.temperature)
        self.mapmaker.run(self.file_save +'/noisy_' )

        #visualize results
        # self._analysis()

    def _run_atmos(self):
        self.lam = models.LinearAngularModel(self.array, 
                                             self.pointing, 
                                             self.site, 
                                             verbose=self.verbose)

        self.lam.simulate_temperature(nu=np.unique(self.array.bands), units='K_RJ')

    def _get_skyconfig(self, **kwargs):
        hudl = fits.open(self.file_name)
        self.im = hudl[0].data
        self.he = hudl[0].header

        if len(self.im.shape) == 4:
            self.im = self.im[0]
        elif len(self.im.shape) == 2:
            self.im = self.im.reshape(1,self.im.shape[0], self.im.shape[1]) * np.ones(len(np.unique(self.array.bands)), self.im.shape[-2], self.im.shape[-1])

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

        #loop here ....
        if self.sky_data['units'] == 'Jy/pixel':
           for idx, uband in enumerate(np.unique(self.array.bands)):
                self.im[idx] = self.im[idx]/utils.KbrightToJyPix(uband, self.sky_data['incell'], self.sky_data['incell'])


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

        for i, uband in enumerate(np.unique(self.array.bands)):

            CMB_map = nmt.synfast_flat(
                nx,
                ny,
                Lx,
                Ly,
                np.array([self.CMB_PS[bandnumber]]),
                [0],
                beam=None,
                seed=self.pointing.seed,
            )[0]

            CMB_map += utils.Tcmb
            CMB_map *= utils.KcmbToKbright(uband) 

            self.im[i] += CMB_map
        
    def _analysis(self):

        if not os.path.exists(self.file_save + "/analyzes"):
            os.mkdir(self.file_save + "/analyzes")

        # visualize scanning patern
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=256, tight_layout=True)
        axes[0].plot(np.degrees(self.lam.pointing.az), np.degrees(self.lam.pointing.el), lw=5e-1)
        axes[0].set_xlabel("az (deg)"), axes[0].set_ylabel("el (deg)")
        axes[1].plot(np.degrees(self.lam.pointing.ra), np.degrees(self.lam.pointing.dec), lw=5e-1)
        axes[1].set_xlabel("ra (deg)"), axes[1].set_ylabel("dec (deg)")
        plt.savefig(self.file_save + "/analyzes/scanpattern_" + self.file_name.replace(".fits", "").split("/")[-1] + ".png")
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
        plt.savefig(self.file_save + "/analyzes/Noise_ps_" + self.file_name.replace(".fits", "").split("/")[-1] + ".png")
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

        plt.savefig(self.file_save + "/analyzes/maps_" + self.file_name.replace(".fits", "").split("/")[-1] + ".png")
        plt.close()