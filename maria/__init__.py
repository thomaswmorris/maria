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
import time

import weathergen
from tqdm import tqdm

import warnings
import healpy as hp

from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from astropy.io import fits

here, this_filename = os.path.split(__file__)
supported_regions = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
regions = weathergen.regions.loc[supported_regions].sort_index()

# -- Specific packages --
from . import utils

with open(f'{here}/configs/arrays.json', 'r+') as f:
    ARRAY_CONFIGS = json.load(f)

with open(f'{here}/configs/pointings.json', 'r+') as f:
    POINTING_CONFIGS = json.load(f)

with open(f'{here}/configs/sites.json', 'r+') as f:
    SITE_CONFIGS = json.load(f)

ARRAYS = list((ARRAY_CONFIGS.keys()))
POINTINGS = list((POINTING_CONFIGS.keys()))
SITES = list((SITE_CONFIGS.keys()))

class InvalidArrayError(Exception):
    def __init__(self, invalid_array):
        super().__init__(f"The array \'{invalid_array}\' is not in the database of default arrays. "
        f"Default arrays are:\n\n{sorted(list(ARRAY_CONFIGS.keys()))}")

class InvalidPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(f"The site \'{invalid_pointing}\' is not in the database of default pointings. "
        f"Default pointings are:\n\n{sorted(list(POINTING_CONFIGS.keys()))}")

class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(f"The site \'{invalid_site}\' is not in the database of default sites. "
        f"Default sites are:\n\n{sorted(list(SITE_CONFIGS.keys()))}")

class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        region_string = regions.to_string(columns=['location', 'country', 'latitude', 'longitude', 'altitude'])
        super().__init__(f"The region \'{invalid_region}\' is not supported. Supported regions are:\n\n{region_string}")


def get_array_config(array_name, **kwargs):
    if not array_name in ARRAY_CONFIGS.keys():
        raise InvalidArrayError(array_name)
    ARRAY_CONFIG = ARRAY_CONFIGS[array_name].copy()
    for k, v in kwargs.items():
        ARRAY_CONFIG[k] = v
    return ARRAY_CONFIG


def get_pointing_config(pointing_name, **kwargs):
    if not pointing_name in POINTING_CONFIGS.keys():
        raise InvalidPointingError(pointing_name)
    POINTING_CONFIG = POINTING_CONFIGS[pointing_name].copy()
    for k, v in kwargs.items():
        POINTING_CONFIG[k] = v
    return POINTING_CONFIG


def get_site_config(site_name, **kwargs):
    if not site_name in SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    SITE_CONFIG = SITE_CONFIGS[site_name].copy()
    for k, v in kwargs.items():
        SITE_CONFIG[k] = v
    return SITE_CONFIG

def get_array(array_name, **kwargs):
    return Array(**get_array_config(array_name, **kwargs))

def get_pointing(pointing_name, **kwargs):
    return Pointing(**get_pointing_config(pointing_name, **kwargs))

def get_site(site_name, **kwargs):
    return Site(**get_site_config(site_name, **kwargs))


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

        _unix = np.atleast_2d(unix).copy()
        _phi = np.atleast_2d(phi).copy()
        _theta = np.atleast_2d(theta).copy()

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
        
        if type(self.detectors) == pd.DataFrame:

            self.offset_x  = self.detectors.offset_x.values
            self.offset_y  = self.detectors.offset_y.values
            self.band      = self.detectors.band.values
            self.bandwidth = self.detectors.bandwidth.values
            self.offsets   = np.c_[self.offset_x, self.offset_y]
            self.n_det     = len(self.detectors)
                
        else:    
            self.band, self.bandwidth, self.n_det = np.empty(0), np.empty(0), 0
            for nu_0, nu_w, n in self.detectors:
                self.band, self.bandwidth = (
                    np.r_[self.band, np.repeat(nu_0, n)],
                    np.r_[self.bandwidth, np.repeat(nu_w, n)],
                )
                self.n_det += n

            self.offsets  = utils.make_array(self.geometry, self.field_of_view, self.n_det)
            self.offsets *= np.pi / 180  # put these in radians

            # scramble up the locations of the bands
            if self.band_grouping == "random":
                random_index = np.random.choice(np.arange(self.n_det), self.n_det, replace=False)
                self.offsets = self.offsets[random_index]

            self.offset_x, self.offset_y = self.offsets.T
            self.r, self.p = np.sqrt(np.square(self.offsets).sum(axis=1)), np.arctan2(*self.offsets.T)

        
        self.ubands = np.unique(self.band)

        # compute detector offsets
        self.hull = sp.spatial.ConvexHull(self.offsets)

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

    @staticmethod
    def validate_pointing_kwargs(kwargs):
        """
        Make sure that we have all the ingredients to produce the pointing data.
        """
        if ('end_time' not in kwargs.keys()) and ('integration_time' not in kwargs.keys()):
            raise ValueError('One of "end_time" or "integration_time" must be in the pointing kwargs.')

    def __init__(self, **kwargs):

        # these are all required kwargs. if they aren't in the passed kwargs, get them from here.
        DEFAULT_POINTING_CONFIG = {
        "integration_time": 600,
        "coord_center": [0, 90],
        "coord_frame": "az_el",
        "scan_pattern": "back-and-forth",  
        "scan_period": 60,  
        "sample_rate": 20, 
        }

        for key, val in kwargs.items():
            setattr(self, key, val)

        for key, val in DEFAULT_POINTING_CONFIG.items():
            if not key in kwargs.keys():
                setattr(self, key, val)

        #self.validate_pointing_kwargs(kwargs)

        # make sure that self.start_datetime exists, and that it's a datetime.datetime object
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        self.start_datetime = utils.datetime_handler(self.start_time)

        # make self.end_datetime
        if hasattr(self, 'end_time'): 
            self.end_datetime = utils.datetime_handler(self.end_time)
        else:
            self.end_datetime = self.start_datetime + timedelta(seconds=self.integration_time)
        
        self.unix_min = self.start_datetime.timestamp()
        self.unix_max = self.end_datetime.timestamp()
        self.dt = 1 / self.sample_rate

        if self.coord_units == "degrees":
            self.coord_center = np.radians(self.coord_center)
            self.coord_throws = np.radians(self.coord_throws)

        self.unix = np.arange(self.unix_min, self.unix_max, self.dt)
        self.n_t = len(self.unix)

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

    def __init__(self, **kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)

        if not self.region in regions.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is not None:
            self.longitude = regions.loc[self.region].longitude

        if self.latitude is not None:
            self.latitude = regions.loc[self.region].latitude

        if self.altitude is not None:
            self.altitude = regions.loc[self.region].altitude

        
