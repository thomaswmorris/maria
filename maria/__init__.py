# -- General packages --
import os
import numpy as np
import scipy as sp

import astropy as ap

import pandas as pd
import h5py
import glob
import re

import weathergen
from tqdm import tqdm

import warnings
import healpy as hp

try:
    import camb
except Exception as e:
    warnings.warn(str(e))
    # warnings.warn(f'Could not import CAMB')

try:
    import pymaster as nmt
except Exception as e:
    warnings.warn(str(e))
    # warnings.warn(f'Could not import namaster')

from datetime import datetime
from matplotlib import pyplot as plt
from astropy.io import fits


# how do we do the bands? this is a great question.
# because all practical telescope instrumentation assume a constant band


# AVE MARIA, GRATIA PLENA, DOMINUS TECUM


here, this_filename = os.path.split(__file__)

# there are a few ways to define an array.
# first, we need to determine the bands:

# the AUTO method requires you to pass:
#
# 'bands' is a list of bands genuses in Hz (e.g. [90e9, 150e9, 220e9])
# 'band_widths' has the same shape and determines the FWHM of the band
# 'dets_per_band' determines how many detectors there will be per band


supported_regions = [re.findall(rf"{here}/am/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/am/*.h5")]
regions = weathergen.regions.loc[supported_regions].sort_index()

# -- Specific packages --
from . import utils
from .configs import *

# -- Don't know what to do with this --

# the 'site' inherits from the weathergen site
DEFAULT_PLAN_CONFIG = {
    "start_time": "2022-07-01T08:00:00",
    "end_time": "2022-07-01T08:10:00",
    "scan_pattern": "daisy",  # [.]   the type of scan strategy (SS)
    "scan_options": {"k": 3.1416},  #
    "coord_center": (45, 45),
    "coord_throw": (2, 2),
    "coord_frame": "azel",
    "scan_period": 120,  # [s]   how often the scan pattern repeats
    "sample_rate": 20,  # [Hz]  how fast to sample
}




class Weobserve:
    def __init__(self, project, skymodel, verbose=True, **kwargs):

        self.verbose = verbose
        self.file_name = skymodel
        self.file_save = project

        # get proper telescope
        self._get_configs(**kwargs)

        # get the atmosphere --> Should do something with the pwv
        self._run_atmos()

        # get the CMB?
        self._get_CMBPS()

        # Get the astronomical signal
        self._get_skyconfig(**kwargs)
        self._get_sky()
        self._savesky()

    def _get_configs(self, **kwargs):

        self.meta_data = {
            "Observatory": kwargs.get("Observatory", "AtLAST"),
            "Scanning_patern": kwargs.get("Scanning_patern", "daisy"),
        }

        # get your defaults
        self.ARRAY_CONFIG = OBSERVATORIES[self.meta_data["Observatory"]]
        self.PLAN_CONFIG = SCANNINGPATTERNS[self.meta_data["Scanning_patern"]]

        # additional telescope request like:
        for k in self.ARRAY_CONFIG.keys():
            if k in kwargs:
                self.ARRAY_CONFIG[k] = kwargs.get(k)

        # additional observational request like:
        for k in self.PLAN_CONFIG.keys():
            if k in kwargs:
                self.PLAN_CONFIG[k] = kwargs.get(k)

        #   integration time --> integration
        #   pwv --> 0.5 mm

    def _run_atmos(self):
        self.lam = LAM(Array(self.ARRAY_CONFIG), Plan(self.PLAN_CONFIG), verbose=self.verbose)

        self.lam.simulate_atmosphere()

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
            (map_data + self.lam.atm_power + cmb_data).ravel(),
            statistic="mean",
            bins=(x_bins, y_bins),
        )[0]

        noise_map = sp.stats.binned_statistic_2d(
            lam_x.ravel(),
            lam_y.ravel(),
            (self.lam.atm_power + cmb_data).ravel(),
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
        f, ps = sp.signal.periodogram(self.lam.atm_power, fs=self.lam.plan.sample_rate, window="tukey")
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


class AtmosphericSpectrum:
    def __init__(self, filepath):
        """
        A class to hold spectra as attributes
        """
        with h5py.File(filepath, "r") as f:
            self.nu = f["nu"][:]  # frequency axis of the spectrum, in GHz
            self.tcwv = f["tcwv"][:]  # total column water vapor, in mm
            self.elev = f["elev"][:]  # elevation, in degrees
            self.t_rj = f["t_rj"][:]  # Rayleigh-Jeans temperature, in Kelvin


class Coordinator:

    # what three-dimensional rotation matrix takes (frame 1) to (frame 2) ?
    # we use astropy to compute this for a few test points, and then use the answer it to efficiently broadcast very big arrays

    def __init__(self, lon, lat):
        self.lc = ap.coordinates.EarthLocation.from_geodetic(lon=lon, lat=lat)

        self.fid_p = np.radians(np.array([0, 0, 90]))
        self.fid_t = np.radians(np.array([90, 0, 0]))
        self.fid_xyz = np.c_[
            np.sin(self.fid_p) * np.cos(self.fid_t),
            np.cos(self.fid_p) * np.cos(self.fid_t),
            np.sin(self.fid_t),
        ]  # the XYZ coordinates of our fiducial test points on the unit sphere

        # in order for this to be efficient, we need to use time-invariant frames
        #

        # you are standing a the north pole looking toward lon = -90 (+x)
        # you are standing a the north pole looking toward lon = 0 (+y)
        # you are standing a the north pole looking up (+z)

    def transform(self, unix, phi, theta, in_frame, out_frame):

        _unix = np.atleast_2d(unix)
        _phi = np.atleast_2d(phi)
        _theta = np.atleast_2d(theta)

        if not _phi.shape == _theta.shape:
            raise ValueError("'phi' and 'theta' must be the same shape")
        if not 1 <= len(_phi.shape) == len(_theta.shape) <= 2:
            raise ValueError("'phi' and 'theta' must be either 1- or 2-dimensional")
        if not unix.shape[-1] == _phi.shape[-1] == _theta.shape[-1]:
            ("'unix', 'phi' and 'theta' must have the same shape in their last axis")

        epoch = _unix.mean()
        obstime = ap.time.Time(epoch, format="unix")
        rad = ap.units.rad

        if in_frame == "az_el":
            self.c = ap.coordinates.SkyCoord(
                az=self.fid_p * rad,
                alt=self.fid_t * rad,
                obstime=obstime,
                frame="altaz",
                location=self.lc,
            )
        if in_frame == "ra_dec":
            self.c = ap.coordinates.SkyCoord(
                ra=self.fid_p * rad,
                dec=self.fid_t * rad,
                obstime=obstime,
                frame="icrs",
                location=self.lc,
            )
        # if in_frame == 'galactic': self.c = ap.coordinates.SkyCoord(l  = self.fid_p * rad, b   = self.fid_t * rad, obstime = ot, frame = 'galactic', location = self.lc)

        if out_frame == "ra_dec":
            self._c = self.c.icrs
            self.rot_p, self.rot_t = self._c.ra.rad, self._c.dec.rad
        if out_frame == "az_el":
            self._c = self.c.altaz
            self.rot_p, self.rot_t = self._c.az.rad, self._c.alt.rad
        # if out_frame == 'galactic': self._c = self.c.galactic; self.rot_p, self.rot_t = self._c.l.rad,  self._c.b.rad

        self.rot_xyz = np.c_[
            np.sin(self.rot_p) * np.cos(self.rot_t),
            np.cos(self.rot_p) * np.cos(self.rot_t),
            np.sin(self.rot_t),
        ]  # the XYZ coordinates of our rotated test points on the unit sphere

        self.R = np.linalg.lstsq(self.fid_xyz, self.rot_xyz, rcond=None)[
            0
        ]  # what matrix takes us (fid_xyz -> rot_xyz)?

        if (in_frame, out_frame) == ("ra_dec", "az_el"):
            _phi -= (_unix - epoch) * (2 * np.pi / 86163.0905)

        trans_xyz = np.swapaxes(
            np.matmul(
                np.swapaxes(
                    np.concatenate(
                        [
                            (np.sin(_phi) * np.cos(_theta))[None],
                            (np.cos(_phi) * np.cos(_theta))[None],
                            np.sin(_theta)[None],
                        ],
                        axis=0,
                    ),
                    0,
                    -1,
                ),
                self.R,
            ),
            0,
            -1,
        )

        trans_phi, trans_theta = np.arctan2(trans_xyz[0], trans_xyz[1]), np.arcsin(trans_xyz[2])

        if (in_frame, out_frame) == ("az_el", "ra_dec"):
            trans_phi += (_unix - epoch) * (2 * np.pi / 86163.0905)

        return np.reshape(trans_phi % (2 * np.pi), phi.shape), np.reshape(trans_theta, theta.shape)


class Array:
    def __init__(self, config, verbose=False):

        self.config = config

        for k, v in config.items():
            setattr(self, k, v)

        self.field_of_view = self.config["field_of_view"]
        self.primary_size = self.config["primary_size"]

        self.bands, self.bandwidths, self.n_det = np.empty(0), np.empty(0), 0
        for nu_0, nu_w, n in self.config["bands"]:
            self.bands, self.bandwidths = (
                np.r_[self.bands, np.repeat(nu_0, n)],
                np.r_[self.bandwidths, np.repeat(nu_w, n)],
            )
            self.n_det += n

        self.offsets = utils.make_array(self.config["geometry"], self.field_of_view, self.n_det)
        self.offsets *= np.pi / 180  # put these in radians

        # compute detector offsets
        self.hull = sp.spatial.ConvexHull(self.offsets)

        # scramble up the locations of the bands
        if self.config["band_grouping"] == "random":
            random_index = np.random.choice(np.arange(self.n_det), self.n_det, replace=False)
            self.offsets = self.offsets[random_index]

        self.offset_x, self.offset_y = self.offsets.T
        self.r, self.p = np.sqrt(np.square(self.offsets).sum(axis=1)), np.arctan2(*self.offsets.T)

        self.ubands = np.unique(self.bands)
        self.nu = np.arange(0, 1e12, 1e9)

        self.passbands = np.c_[
            [utils.get_passband(self.nu, nu_0, nu_w, order=16) for nu_0, nu_w in zip(self.bands, self.bandwidths)]
        ]

        nu_mask = (self.passbands > 1e-4).any(axis=0)
        self.nu, self.passbands = self.nu[nu_mask], self.passbands[:, nu_mask]

        self.passbands /= self.passbands.sum(axis=1)[:, None]

        # compute beams
        self.optical_model = "diff_lim"
        if self.optical_model == "diff_lim":

            self.get_beam_waist = lambda z, w_0, f: w_0 * np.sqrt(
                1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0))
            )
            self.get_beam_profile = lambda r, r_fwhm: np.exp(np.log(0.5) * np.abs(r / r_fwhm) ** 8)
            self.beam_func = self.get_beam_profile

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


class Pointing:

    """
    A class containing time-ordered pointing data.
    """

    def __init__(self, config, verbose=False):

        self.config = config
        for key, val in config.items():
            setattr(self, key, val)
            if verbose:
                print(f"set {key} to {val}")

        self.compute()

    def compute(self):

        self.dt = 1 / self.sample_rate

        self.start_time = utils.datetime_handler(self.start_time)
        self.end_time = utils.datetime_handler(self.end_time)

        self.t_min = self.start_time.timestamp()
        self.t_max = self.end_time.timestamp()

        if self.coord_units == "degrees":
            self.coord_center = np.radians(self.coord_center)
            self.coord_throws = np.radians(self.coord_throws)

        self.unix = np.arange(self.t_min, self.t_max, self.dt)
        self.coords = utils.get_pointing(
            self.unix,
            self.scan_period,
            self.coord_center,
            self.coord_throws,
            self.scan_pattern,
        )

        if self.coord_frame == "ra_dec":
            self.ra, self.dec = self.coords

        if self.coord_frame == "az_el":
            self.az, self.el = self.coords

        if self.coord_frame == "dx_dy":
            self.dx, self.dy = self.coords


class Site:

    """
    A class containing time-ordered pointing data. Pass a supported site (found at weathergen.sites),
    and a height correction if needed.
    """

    def __init__(self, region=None, latitude=None, longitude=None, altitude=None):

        self.region = region

        self.longitude = longitude if longitude is not None else weathergen.regions.loc[region].longitude
        self.latitude = latitude if latitude is not None else weathergen.regions.loc[region].latitude
        self.altitude = altitude if altitude is not None else weathergen.regions.loc[region].altitude


class AtmosphericModel:
    """
    The base class for modeling atmospheric fluctuations.

    A model needs to have the functionality to generate spectra for any pointing data we supply it with.


    """

    def __init__(self, array, pointing, site):

        self.array, self.pointing, self.site = array, pointing, site
        self.spectrum = AtmosphericSpectrum(filepath=f"{here}/am/{self.site.region}.h5")
        self.coordinator = Coordinator(lat=self.site.latitude, lon=self.site.longitude)

        if self.pointing.coord_frame == "az_el":
            self.pointing.ra, self.pointing.dec = self.coordinator.transform(
                self.pointing.unix,
                self.pointing.az,
                self.pointing.el,
                in_frame="az_el",
                out_frame="ra_dec",
            )
            self.pointing.dx, self.pointing.dy = utils.to_xy(
                self.pointing.az,
                self.pointing.el,
                self.pointing.az.mean(),
                self.pointing.el.mean(),
            )

        if self.pointing.coord_frame == "ra_dec":
            self.pointing.az, self.pointing.el = self.coordinator.transform(
                self.pointing.unix,
                self.pointing.ra,
                self.pointing.dec,
                in_frame="ra_dec",
                out_frame="az_el",
            )
            self.pointing.dx, self.pointing.dy = utils.to_xy(
                self.pointing.az,
                self.pointing.el,
                self.pointing.az.mean(),
                self.pointing.el.mean(),
            )

        self.azim, self.elev = utils.from_xy(
            self.array.offset_x[:, None],
            self.array.offset_y[:, None],
            self.pointing.az,
            self.pointing.el,
        )

        if self.elev.min() < np.radians(10):
            warnings.warn(
                f"Some detectors come within 10 degrees of the horizon (el_min = {np.degrees(self.elev.min()):.01f}°)"
            )
        if self.elev.min() <= 0:
            raise PointingError(
                f"Some detectors are pointing below the horizon (el_min = {np.degrees(self.elev.min()):.01f}°). Please refer to:"
                "https://1.bp.blogspot.com/-dXMlsHE-rUI/UbWXQcc8aVI/AAAAAAAAEHw/fHwfk_zjVNQ/s1600"
            )


class ArrayError(Exception):
    pass


class PointingError(Exception):
    pass


class SiteError(Exception):
    pass
