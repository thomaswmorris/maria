# -- General packages --
import os
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt
from astropy.io import fits

from . import atmosphere, get_array, get_site, get_pointing
from . import utils

class WeObserve:
    def __init__(self, project, skymodel, array_name='AtLAST', pointing_name='DAISY_2deg_4ra_10.5dec_600s', site_name='APEX', verbose=True, cmb = False, **kwargs):

        self.verbose   = verbose
        self.file_name = skymodel
        self.file_save = project
        self.add_cmb   = cmb

        self.array    = get_array(array_name, **kwargs)
        self.pointing = get_pointing(pointing_name, **kwargs)
        self.site     = get_site(site_name, **kwargs)

        # get the atmosphere --> Should do something with the pwv
        self._run_atmos()

        # get the CMB
        if self.add_cmb:
            self._get_CMBPS()

        # Get the astronomical signal
        self._get_skyconfig(**kwargs)
        self._get_sky()
        self._savesky()
        

    def _savesky(
        self,
    ):
        if not os.path.exists(self.file_save):
            os.mkdir(self.file_save)

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