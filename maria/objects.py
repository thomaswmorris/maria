# import numpy as np
# import scipy as sp
# import pandas as pd

# import glob
# import os
# import re
# import weathergen

# from datetime import datetime, timedelta

# from . import utils

# here, this_filename = os.path.split(__file__)
# supported_regions = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
# regions = weathergen.regions.loc[supported_regions].sort_index()

# class InvalidRegionError(Exception):
#     def __init__(self, invalid_region):
#         region_string = regions.to_string(columns=['location', 'country', 'latitude', 'longitude', 'altitude'])
#         super().__init__(f"The region \'{invalid_region}\' is not supported. Supported regions are:\n\n{region_string}")

# class Array:
#     def __init__(self, **kwargs):

#         DEFAULT_ARRAY_CONFIG = {
#         "detectors": [
#             [150e9, 10e9, 100], 
#         ],
#         "geometry": "hex", 
#         "field_of_view": 1.3, 
#         "primary_size": 50,   
#         "band_grouping": "randomized",   
#         "az_bounds": [0, 360],  
#         "el_bounds": [20, 90],
#         "max_az_vel": 3,
#         "max_el_vel": 2,
#         "max_az_acc": 1,
#         "max_el_acc": 0.25
#         }

#         for k, v in kwargs.items():
#             setattr(self, k, v)
        
#         if type(self.detectors) == pd.DataFrame:

#             self.offset_x    = self.detectors.offset_x.values
#             self.offset_y    = self.detectors.offset_y.values
#             self.band_center = self.detectors.band.values
#             self.band_width  = self.detectors.bandwidth.values
#             self.offset      = np.c_[self.offset_x, self.offset_y]
#             self.n_det       = len(self.detectors)
                
#         else:    
#             self.band_center, self.band_width, self.n_det = np.empty(0), np.empty(0), 0
#             for nu_0, nu_w, n in self.detectors:
#                 self.band_center, self.band_width = (
#                     np.r_[self.band_center, np.repeat(nu_0, n)],
#                     np.r_[self.band_width, np.repeat(nu_w, n)],
#                 )
#                 self.n_det += n

#             self.offset  = utils.make_array(self.geometry, self.field_of_view, self.n_det)
#             self.offset *= np.pi / 180  # put these in radians

#             # scramble up the locations of the bands
#             if self.band_grouping == "randomized":
#                 random_index = np.random.choice(np.arange(self.n_det), self.n_det, replace=False)
#                 self.offset = self.offset[random_index]

#             self.offset_x, self.offset_y = self.offset.T
#             self.r, self.p = np.sqrt(np.square(self.offset).sum(axis=1)), np.arctan2(*self.offset.T)

        
#         self.band = np.array([f"f{int(nu/10**(3*int(np.log10(nu)/3))):03}" for nu in self.band_center])

#         self.ubands = np.unique(self.band)

#         # compute detector offset
#         self.hull = sp.spatial.ConvexHull(self.offset)

#         # compute beams
#         self.optical_model = "diff_lim"
#         if self.optical_model == "diff_lim":

#             self.get_beam_waist = lambda z, w_0, f: w_0 * np.sqrt(
#                 1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0))
#             )
#             self.get_beam_profile = lambda r, r_fwhm: np.exp(np.log(0.5) * np.abs(r / r_fwhm) ** 8)
#             self.beam_func = self.get_beam_profile

        
#         self.metadata = pd.DataFrame(index=np.arange(self.n_det))

#         self.metadata.loc[:, "band"]     = self.band
#         self.metadata.loc[:, "nom_freq"] = self.band_center.astype(int)
#         self.metadata.loc[:, "offset_x"] = self.offset_x
#         self.metadata.loc[:, "offset_y"] = self.offset_y


#     def make_filter(self, waist, res, func, width_per_waist=1.2):

#         filter_width = width_per_waist * waist
#         n_filter = 2 * int(np.ceil(0.5 * filter_width / res)) + 1

#         filter_side = 0.5 * np.linspace(-filter_width, filter_width, n_filter)

#         FILTER_X, FILTER_Y = np.meshgrid(filter_side, filter_side, indexing="ij")
#         FILTER_R = np.sqrt(np.square(FILTER_X) + np.square(FILTER_Y))

#         FILTER = func(FILTER_R, 0.5 * waist)
#         FILTER /= FILTER.sum()

#         return FILTER

#     def separate_filter(self, F, tol=1e-2):

#         u, s, v = np.linalg.svd(F)
#         eff_filter = 0
#         for m, (_u, _s, _v) in enumerate(zip(u.T, s, v)):

#             eff_filter += _s * np.outer(_u, _v)
#             if np.abs(F - eff_filter).sum() < tol:
#                 break

#         return u.T[: m + 1], s[: m + 1], v[: m + 1]

#     def separably_filter(self, M, F, tol=1e-2):

#         u, s, v = self.separate_filter(F, tol=tol)

#         filt_M = 0
#         for _u, _s, _v in zip(u, s, v):

#             filt_M += _s * sp.ndimage.convolve1d(sp.ndimage.convolve1d(M.astype(float), _u, axis=0), _v, axis=1)

#         return filt_M



# class Pointing:

#     """
#     A class containing time-ordered pointing data.
#     """

#     @staticmethod
#     def validate_pointing_kwargs(kwargs):
#         """
#         Make sure that we have all the ingredients to produce the pointing data.
#         """
#         if ('end_time' not in kwargs.keys()) and ('integration_time' not in kwargs.keys()):
#             raise ValueError('One of "end_time" or "integration_time" must be in the pointing kwargs.')

#     def __init__(self, **kwargs):

#         # these are all required kwargs. if they aren't in the passed kwargs, get them from here.
#         DEFAULT_POINTING_CONFIG = {
#         "integration_time": 600,
#         "scan_center": [0, 90],
#         "pointing_frame": "az_el",
#         "pointing_units": "degrees",
#         "scan_pattern": "back-and-forth",  
#         "scan_period": 60,  
#         "sample_rate": 20, 
#         }

#         for key, val in kwargs.items():
#             setattr(self, key, val)

#         for key, val in DEFAULT_POINTING_CONFIG.items():
#             if not key in kwargs.keys():
#                 setattr(self, key, val)

#         #self.validate_pointing_kwargs(kwargs)

#         # make sure that self.start_datetime exists, and that it's a datetime.datetime object
#         if not hasattr(self, 'start_time'):
#             self.start_time = datetime.now().timestamp()
#         self.start_datetime = utils.datetime_handler(self.start_time)

#         # make self.end_datetime
#         if hasattr(self, 'end_time'): 
#             self.end_datetime = utils.datetime_handler(self.end_time)
#         else:
#             self.end_datetime = self.start_datetime + timedelta(seconds=self.integration_time)
        
#         self.time_min = self.start_datetime.timestamp()
#         self.time_max = self.end_datetime.timestamp()
#         self.dt = 1 / self.sample_rate

#         if self.pointing_units == "degrees":
#             self.scan_center = np.radians(self.scan_center)

#         self.time = np.arange(self.time_min, self.time_max, self.dt)
#         self.n_time = len(self.time)

#         time_ordered_pointing = utils.get_pointing(
#             self.time,
#             self.scan_period,
#             self.scan_center,
#             self.pointing_throws,
#             self.scan_pattern,
#         )

#         if self.pointing_frame == "ra_dec":
#             self.ra, self.dec = time_ordered_pointing
#         elif self.pointing_frame == "az_el":
#             self.az, self.el = time_ordered_pointing
#         elif self.pointing_frame == "dx_dy":
#             self.dx, self.dy = time_ordered_pointing


# class Site:

#     """
#     A class containing time-ordered pointing data. Pass a supported site (found at weathergen.sites),
#     and a height correction if needed.
#     """

#     def __init__(self, **kwargs):

#         for key, val in kwargs.items():
#             setattr(self, key, val)

#         if not self.region in regions.index.values:
#             raise InvalidRegionError(self.region)

#         if self.longitude is None:
#             self.longitude = regions.loc[self.region].longitude

#         if self.latitude is None:
#             self.latitude = regions.loc[self.region].latitude

#         if self.altitude is None:
#             self.altitude = regions.loc[self.region].altitude

        
