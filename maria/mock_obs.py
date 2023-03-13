# -- General packages --
import os
import numpy as np
import scipy as sp

import camb
import pymaster as nmt

from matplotlib import pyplot as plt
from astropy.io import fits

from . import get_array, get_site, get_pointing
from . import models, utils

class WeObserve:
    def __init__(self, project, skymodel, array_name='AtLAST', pointing_name='DAISY_2deg_4ra_10.5dec_600s', site_name='APEX', verbose=True, **kwargs):

        self.verbose = verbose
        self.file_name = skymodel
        self.file_save = project

        self.array = get_array(array_name)
        self.pointing = get_pointing(pointing_name)
        self.site = get_site(site_name)

        # get the atmosphere --> Should do something with the pwv
        self._run_atmos()

        # get the CMB?
        self._get_CMBPS()

        # Get the astronomical signal
        self._get_skyconfig(**kwargs)
        self._get_sky()
        self._savesky()


    def _run_atmos(self):

        self.lam = models.LinearAngularModel(self.array, self.pointing, self.site, verbose=self.verbose)
        self.lam.simulate_temperature_rayleigh_jeans()

    def _get_CMBPS(
        self,
    ):

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(5000, lens_potential_accuracy=0)

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="K")["total"][:, 0]

        # HMMMM there is a frequency dependance
        self.CMB_PS = np.empty((len(self.ARRAY_CONFIG["bands"]), len(powers)))
        for i in range(len(self.ARRAY_CONFIG["bands"])):
            self.CMB_PS[i] = powers

    def _cmb_imager(self, bandnumber=0):

        nx, ny = self.im.shape
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
            seed=self.PLAN_CONFIG["seed"],
        )[0]

    def _get_skyconfig(self, **kwargs):
        hudl = fits.open(self.file_name)
        self.im = hudl[0].data
        self.he = hudl[0].header

        self.sky_data = {
            "inbright": kwargs.get("inbright", None),  # assuming something: Jy/pix?
            "incell": kwargs.get("incell", self.he["CDELT1"]),  # assuming written in degree
            "inwidth": kwargs.get("inwidth", None),  # assuming written in Hz --> for the spectograph...
        }

        if self.sky_data["inbright"] != None:
            self.im = self.im / np.nanmax(self.im) * self.sky_data["inbright"]

    # need to rewrite this
    def _get_sky(
        self,
    ):

        map_res = np.radians(self.sky_data["incell"])
        map_nx, map_ny = self.im.shape

        map_x = map_res * map_nx * np.linspace(-0.5, 0.5, map_nx)
        map_y = map_res * map_ny * np.linspace(-0.5, 0.5, map_ny)

        map_X, map_Y = np.meshgrid(map_x, map_y, indexing="ij")
        map_azim, map_elev = utils.from_xy(map_X, map_Y, self.lam.azim.mean(), self.lam.elev.mean())

        lam_x, lam_y = utils.to_xy(self.lam.elev, self.lam.azim, self.lam.elev.mean(), self.lam.azim.mean())

        # MAP  MAKING STUFF
        map_data = sp.interpolate.RegularGridInterpolator(
            (map_x, map_y), self.im, bounds_error=False, fill_value=0
        )((lam_x, lam_y))

        self._cmb_imager()
        cmb_data = sp.interpolate.RegularGridInterpolator(
            (map_x, map_y), self.CMB_map, bounds_error=False, fill_value=0
        )((lam_x, lam_y))

        x_bins = np.arange(map_X.min(), map_X.max(), 8 * map_res)
        y_bins = np.arange(map_Y.min(), map_Y.max(), 8 * map_res)

        true_map = sp.stats.binned_statistic_2d(
            map_X.ravel(),
            map_Y.ravel(),
            self.im.ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        filtered_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            map_data.ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        total_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            (map_data + self.lam.temperature_rayleigh_jeans + cmb_data).ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        noise_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            (self.lam.temperature_rayleigh_jeans + cmb_data).ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        self.truesky = true_map
        self.noisemap = noise_map
        self.filteredmap = filtered_map
        self.mockobs = total_map

    def _savesky(
        self,
    ):

        if not os.path.exists(self.file_save):
            os.mkdir(self.file_save)

        # update header with the kwargs
        fits.writeto(
            self.file_save + "/" + self.file_name.replace(".fits", "_noisemap.fits").split("/")[-1],
            self.noisemap,
            header=self.he,
            overwrite=True,
        )
        fits.writeto(
            self.file_save + "/" + self.file_name.replace(".fits", "_filtered.fits").split("/")[-1],
            self.filteredmap,
            header=self.he,
            overwrite=True,
        )
        fits.writeto(
            self.file_save + "/" + self.file_name.replace(".fits", "_synthetic.fits").split("/")[-1],
            self.mockobs,
            header=self.he,
            overwrite=True,
        )

        if not os.path.exists(self.file_save + "/analyzes"):
            os.mkdir(self.file_save + "/analyzes")

        # visualize scanning patern
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=256, tight_layout=True)
        axes[0].plot(np.degrees(self.lam.c_az), np.degrees(self.lam.c_el), lw=5e-1)
        axes[0].set_xlabel("az (deg)"), axes[0].set_ylabel("el (deg)")
        axes[1].plot(np.degrees(self.lam.c_ra), np.degrees(self.lam.c_dec), lw=5e-1)
        axes[1].set_xlabel("ra (deg)"), axes[1].set_ylabel("dec (deg)")
        plt.savefig(
            self.file_save + "/analyzes/scanpattern_" + self.file_name.replace(".fits", "").split("/")[-1] + ".png"
        )
        plt.close()

        # visualize powerspectrum
        f, ps = sp.signal.periodogram(self.lam.temperature_rayleigh_jeans, fs=self.lam.pointing.sample_rate, window="tukey")
        plt.figure()
        plt.plot(f[1:], ps.mean(axis=0)[1:], label="atmosphere")
        plt.plot(f[1:], f[1:] ** (-8 / 3), label="y = f^-(8/3)")
        plt.loglog()
        plt.xlabel("l")
        plt.ylabel("PS")
        plt.legend()
        plt.savefig(
            self.file_save + "/analyzes/Noise_ps_" + self.file_name.replace(".fits", "").split("/")[-1] + ".png"
        )
        plt.close()

        # visualize fits files
        fig, (true_ax, signal_ax, noise_ax, total_ax) = plt.subplots(
            1, 4, figsize=(9, 3), sharex=True, sharey=True, constrained_layout=True
        )

        total_plt = true_ax.imshow(self.truesky)
        true_ax.set_title("True map")
        fig.colorbar(total_plt, ax=true_ax, location="bottom", shrink=0.8)

        true_plt = signal_ax.imshow(self.filteredmap)
        signal_ax.set_title("Filtered map")
        fig.colorbar(true_plt, ax=signal_ax, location="bottom", shrink=0.8)

        signal_plt = noise_ax.imshow(self.noisemap)
        noise_ax.set_title("Noise map")
        fig.colorbar(signal_plt, ax=noise_ax, location="bottom", shrink=0.8)

        total_plt = total_ax.imshow(self.mockobs)
        total_ax.set_title("Synthetic Observation")
        fig.colorbar(total_plt, ax=total_ax, location="bottom", shrink=0.8)

        plt.savefig(
            self.file_save + "/analyzes/maps_" + self.file_name.replace(".fits", "").split("/")[-1] + ".png"
        )
        plt.close()