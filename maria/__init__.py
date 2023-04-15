# Ave, María, grátia plena, Dóminus tecum

from ._version import get_versions
__version__ = get_versions()["version"]

del get_versions

import os
import numpy as np
import scipy as sp

import astropy as ap

import pandas as pd
import h5py
import glob
import re
import json

import weathergen
from tqdm import tqdm

import warnings
import healpy as hp

from datetime import datetime
from matplotlib import pyplot as plt
from astropy.io import fits

here, this_filename = os.path.split(__file__)
supported_regions = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
regions = weathergen.regions.loc[supported_regions].sort_index()

# -- Specific packages --
from . import utils

with open(f'{here}/configs/arrays.json', 'r+') as f:
    DEFAULT_ARRAY_CONFIGS = json.load(f)

with open(f'{here}/configs/pointings.json', 'r+') as f:
    DEFAULT_POINTING_CONFIGS = json.load(f)

with open(f'{here}/configs/sites.json', 'r+') as f:
    DEFAULT_SITE_CONFIGS = json.load(f)

DEFAULT_ARRAYS = list((DEFAULT_ARRAY_CONFIGS.keys()))
DEFAULT_POINTINGS = list((DEFAULT_POINTING_CONFIGS.keys()))
DEFAULT_SITES = list((DEFAULT_SITE_CONFIGS.keys()))


class InvalidArrayError(Exception):
    def __init__(self, invalid_array):
        super().__init__(f"The array \'{invalid_array}\' is not in the database of default arrays. "
        f"Default arrays are:\n\n{sorted(list(DEFAULT_ARRAY_CONFIGS.keys()))}")

class InvalidPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(f"The site \'{invalid_pointing}\' is not in the database of default pointings. "
        f"Default pointings are:\n\n{sorted(list(DEFAULT_POINTING_CONFIGS.keys()))}")

class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(f"The site \'{invalid_site}\' is not in the database of default sites. "
        f"Default sites are:\n\n{sorted(list(DEFAULT_SITE_CONFIGS.keys()))}")

class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        region_string = regions.to_string(columns=['location', 'country', 'latitude', 'longitude', 'altitude'])
        super().__init__(f"The region \'{invalid_region}\' is not supported. Supported regions are:\n\n{region_string}")

class PointingError(Exception):
    pass

def validate_pointing(azim, elev):
    el_min = np.atleast_1d(elev).min()
    if el_min < np.radians(10):
        warnings.warn(f"Some detectors come within 10 degrees of the horizon (el_min = {np.degrees(el_min):.01f}°)")
    if el_min <= 0:
        raise PointingError(f"Some detectors are pointing below the horizon (el_min = {np.degrees(el_min):.01f}°)")


def get_array_config(array_name):
    if not array_name in DEFAULT_ARRAY_CONFIGS.keys():
        raise InvalidArrayError(array_name)
    return DEFAULT_ARRAY_CONFIGS[array_name].copy()

def get_pointing_config(pointing_name):
    if not pointing_name in DEFAULT_POINTING_CONFIGS.keys():
        raise InvalidPointingError(pointing_name)
    return DEFAULT_POINTING_CONFIGS[pointing_name].copy()

def get_site_config(site_name):
    if not site_name in DEFAULT_SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    return DEFAULT_SITE_CONFIGS[site_name].copy()

def get_array(array_name):
    return Array(get_array_config(array_name))

def get_pointing(pointing_name):
    return Pointing(get_pointing_config(pointing_name))

def get_site(site_name):
    return Site(get_site_config(site_name))


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

    def __init__(self, config, verbose=False):

        self.seasonal = True
        self.diurnal = True
        self.fixed_quantiles = {}

        self.config = config
        for key, val in config.items():
            setattr(self, key, val)
            if verbose:
                print(f"set {key} to {val}")

        if not self.region in regions.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is not None:
            self.longitude = regions.loc[self.region].longitude

        if self.latitude is not None:
            self.latitude = regions.loc[self.region].latitude

        if self.altitude is not None:
            self.altitude = regions.loc[self.region].altitude

        

class AtmosphericModel:
    """
    The base class for modeling atmospheric fluctuations.

    A model needs to have the functionality to generate spectra for any pointing data we supply it with.


    """

    def __init__(self, array, pointing, site):

        self.array, self.pointing, self.site = array, pointing, site
        self.spectrum = AtmosphericSpectrum(filepath=f"{here}/spectra/{self.site.region}.h5")
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

        validate_pointing(self.azim, self.elev)

        self.weather = weathergen.Weather(
            region=self.site.region,
            seasonal=self.site.seasonal,
            diurnal=self.site.diurnal,
        )


    def simulate_integrated_water_vapor(self):
        raise NotImplementedError('Atmospheric simulations are not implemented in the base class!')


    def simulate_temperature(self, nu=150e9, units='K_RJ'):

        if units == 'K_RJ': # Kelvin Rayleigh-Jeans

            self.simulate_integrated_water_vapor() # this is elevation-corrected by default

            TRJ_interpolator = sp.interpolate.RegularGridInterpolator((self.spectrum.elev, 
                                                                    self.spectrum.tcwv,
                                                                    self.spectrum.nu,),
                                                                    self.spectrum.t_rj)

            self.temperature = TRJ_interpolator((np.degrees(self.elev)[None], 
                                                 self.integrated_water_vapor[None], 
                                                 np.atleast_1d(nu)[:,None,None]))

        if units == 'F_RJ': # Fahrenheit Rayleigh-Jeans (honestly it just feels more natural)

            self.simulate_temperature(self, nu=nu, units='K_RJ')
            self.temperature = 1.8 * (self.temperature - 273.15) + 32


        

