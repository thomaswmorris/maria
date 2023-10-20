import numpy as np
import scipy as sp
import pandas as pd
from operator import attrgetter
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection

from dataclasses import dataclass, fields
from typing import Tuple
from collections.abc import Mapping
import os

from . import utils

# better formatting for pandas dataframes
# pd.set_eng_float_format()

here, this_filename = os.path.split(__file__)

ARRAY_CONFIGS = utils.io.read_yaml(f"{here}/configs/arrays.yml")
ARRAY_PARAMS = set()
for key, config in ARRAY_CONFIGS.items():
    ARRAY_PARAMS |= set(config.keys())

class UnsupportedArrayError(Exception):
    def __init__(self, invalid_array):
        array_df = pd.DataFrame(columns=["description", "documentation"])
        for key, config in ARRAY_CONFIGS.items():
            array_df.loc[key, "description"] = config["description"]
            array_df.loc[key, "documentation"] = config["documentation"]
        super().__init__(f"The array \'{invalid_array}\' is not in the database of default arrays. "
        f"Default arrays are:\n\n{array_df.sort_index()}")

def get_array_config(array_name, **kwargs):
    if not array_name in ARRAY_CONFIGS.keys():
        raise UnsupportedArrayError(array_name)
    ARRAY_CONFIG = ARRAY_CONFIGS[array_name].copy()
    for k, v in kwargs.items():
        ARRAY_CONFIG[k] = v
    return ARRAY_CONFIG

def get_array(array_name, **kwargs):
    """
    Get an array from a pre-defined config.
    """
    return Array.from_config(get_array_config(array_name, **kwargs))


REQUIRED_DET_CONFIG_KEYS = ["n", "band_center", "band_width"]

def generate_dets_from_config(bands: Mapping, field_of_view: float, geometry: str = 'hex', max_baseline: float = 0, randomize_offsets: bool = True):

    dets = pd.DataFrame(columns=["band", "band_center", "band_width", "offset_x", "offset_y", "baseline_x", "baseline_y"], dtype=float)

    for band, band_config in bands.items():
        
        if not all(key in band_config.keys() for key in REQUIRED_DET_CONFIG_KEYS):
            raise ValueError(f'Each band must have keys {REQUIRED_DET_CONFIG_KEYS}')

        band_dets = pd.DataFrame(index=np.arange(band_config["n"]))
        band_dets.loc[:, "band"] = band
        band_dets.loc[:, "band_center"] = band_config["band_center"]
        band_dets.loc[:, "band_width"] = band_config["band_width"]

        dets = pd.concat([dets, band_dets])

    offsets_radians = np.radians(utils.generate_array_offsets(geometry, field_of_view, len(dets)))

    # should we make another function for this?
    baseline = utils.generate_array_offsets(geometry, max_baseline, len(dets))

    if randomize_offsets:
        np.random.shuffle(offsets_radians) # this is a stupid function.

    dets.loc[:, "offset_x"] = offsets_radians[:, 0]
    dets.loc[:, "offset_y"] = offsets_radians[:, 1]
    dets.loc[:, "baseline_x"] = baseline[:, 0]
    dets.loc[:, "baseline_y"] = baseline[:, 1]

    for key in ['offset_x', 'offset_y', 'baseline_x', 'baseline_y']:
        dets.loc[:, key] = dets.loc[:, key].astype(float)

    return dets


@dataclass
class Array:
    """
    An array.
    """
    description: str = ''
    primary_size: float = 5    # in meters
    field_of_view: float = 1   # in deg
    geometry: str = 'hex'
    max_az_vel: float = 0      # in deg/s
    max_el_vel: float = np.inf # in deg/s
    max_az_acc: float = 0      # in deg/s^2
    max_el_acc: float = np.inf # in deg/s^2
    az_bounds: Tuple[float, float] = (0, 360) # in degrees
    el_bounds: Tuple[float, float] = (0, 90)  # in degrees
    documentation: str = ''
    dets: pd.DataFrame = None    # dets, it's complicated

    def __repr__(self):
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self))
            for f in fields(self)
            if f.name != "dets"
        )

        nodef_f_repr = ", ".join(f"{name}={value}" for name, value in nodef_f_vals)
        return f"{self.__class__.__name__}({nodef_f_repr})"

    @property
    def ubands(self):
        return np.unique(self.dets.band)

    @property
    def offset_x(self):
        return self.dets.offset_x.values

    @property
    def offset_y(self):
        return self.dets.offset_y.values

    @property
    def offsets(self):
        return np.c_[self.offset_x, self.offset_y]

    @property
    def baseline_x(self):
        return self.dets.baseline_x.values

    @property
    def baseline_y(self):
        return self.dets.baseline_y.values

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y]


    @classmethod
    def from_config(cls, config):

        if isinstance(config["dets"], Mapping):
            field_of_view = config.get("field_of_view", 1)
            geometry = config.get("geometry", "hex")
            dets = generate_dets_from_config(config["dets"], 
                                            field_of_view=field_of_view,
                                            geometry=geometry)

        return cls(
                description=config["description"],
                primary_size=config["primary_size"],
                field_of_view=field_of_view,
                geometry=geometry,
                max_az_vel=config["max_az_vel"],
                max_el_vel=config["max_el_vel"],
                max_az_acc=config["max_az_acc"],
                max_el_acc=config["max_el_acc"],
                az_bounds=config["az_bounds"],
                el_bounds=config["el_bounds"],
                documentation=config["documentation"],
                dets=dets,
                )

    @staticmethod
    def beam_profile(r, fwhm):
        return np.exp(np.log(0.5) * np.abs(r / fwhm) ** 8)
    
    @property
    def band_min(self):
        return (self.dets.band_center - 0.5 * self.dets.band_width).values

    @property
    def band_max(self):
        return (self.dets.band_center + 0.5 * self.dets.band_width).values

    def passband(self, nu):
        """
        Passband response as a function of nu (in Hz)
        """
        return ((nu[None] > self.band_min[:, None]) & (nu[None] < self.band_max[:, None])).astype(float)

    def angular_fwhm(self, z):
        return utils.gaussian_beam_angular_fwhm(z=z, w_0=self.primary_size/np.sqrt(2*np.log(2)), f=self.dets.band_center.values, n=1)

    def physical_fwhm(self, z):
        return z * self.angular_fwhm(z)

    def angular_beam(self, r, z=np.inf, n=1, l=None, f=None):
        """
        Beam response as a function of radius (in radians)
        """
        return self.beam_profile(r, self.angular_fwhm(z))

    def physical_beam(self, r, z=np.inf, n=1, l=None, f=None):
        """
        Beam response as a function of radius (in meters)
        """
        return self.beam_profile(r, self.physical_fwhm(z))

    @property
    def n_dets(self):
        return len(self.dets)

    @property
    def ubands(self):
        return list(np.unique(self.dets.band))
         
         
    def plot_dets(self):

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=256)

        for uband in self.ubands:

            band_mask = self.dets.band == uband

            band_res = 60 * np.degrees(2.998e8 / (self.dets.band_center.mean() * self.primary_size))
            offsets_arcmins = 60 * np.degrees(self.offsets)

            ax.add_collection(EllipseCollection(widths=band_res, heights=band_res, angles=0, units='xy',
                                            facecolors="none", edgecolors="k", lw=1e-1,
                                            offsets=offsets_arcmins, transOffset=ax.transData))
            
            ax.scatter(*offsets_arcmins.T, label=uband, s=5e-1)

        ax.set_xlabel(r'$\theta_x$ offset (arc min.)')
        ax.set_ylabel(r'$\theta_y$ offset (arc min.)')
        ax.legend()

