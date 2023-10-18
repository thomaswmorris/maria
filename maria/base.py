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

from .array import get_array, ARRAY_PARAMS
from .coordinator import Coordinator
from .pointing import get_pointing, POINTING_PARAMS
from .site import get_site, SITE_PARAMS
from .tod import TOD

class BaseSimulation:
    """
    The base class for a simulation. This is an ingredient in every simulation.
    """

    def __init__(self, 
                 array, 
                 pointing, 
                 site,
                 **kwargs):

        # who does each kwarg belong to?
        array_kwargs = {k:v for k, v in kwargs.items() if k in ARRAY_PARAMS}
        pointing_kwargs = {k:v for k, v in kwargs.items() if k in POINTING_PARAMS}
        site_kwargs = {k:v for k, v in kwargs.items() if k in SITE_PARAMS}

        self.array = get_array(array, **array_kwargs) if isinstance(array, str) else array
        self.pointing = get_pointing(pointing, **pointing_kwargs) if isinstance(pointing, str) else pointing
        self.site = get_site(site, **site_kwargs) if isinstance(site, str) else site

        self.coordinator = Coordinator(lat=self.site.latitude, lon=self.site.longitude)

        if self.pointing.pointing_frame == "az_el":
            self.pointing.ra, self.pointing.dec = self.coordinator.transform(
                self.pointing.time,
                self.pointing.az,
                self.pointing.el,
                in_frame="az_el",
                out_frame="ra_dec",
            )

        if self.pointing.pointing_frame == "ra_dec":
            self.pointing.az, self.pointing.el = self.coordinator.transform(
                self.pointing.time,
                self.pointing.ra,
                self.pointing.dec,
                in_frame="ra_dec",
                out_frame="az_el",
            )

    @property
    def params(self):
        _params = {"array": {}, "pointing": {}, "site": {}}
        for key in ARRAY_PARAMS:
            _params["array"][key] = getattr(self.array, key)
        for key in POINTING_PARAMS:
            _params["pointing"][key] = getattr(self.pointing, key)
        for key in SITE_PARAMS:
            _params["site"][key] = getattr(self.site, key)
        return _params


    def _run(self):

        raise NotImplementedError()


    def run(self):

        self._run()
    
        tod = TOD()

        tod.data = self.data # this should be set in the _run() method

        tod.time = self.pointing.time
        tod.az   = self.pointing.az
        tod.el   = self.pointing.el
        tod.ra   = self.pointing.ra
        tod.dec  = self.pointing.dec
        tod.cntr = self.pointing.scan_center
        
        if hasattr(self, "map_sim"):
            if self.map_sim is not None:
                tod.unit = self.map_sim.input_map.units
                tod.header = self.map_sim.input_map.header
            else:
                tod.unit = 'K'
                tod.header = fits.header.Header()


        tod.dets = self.array.dets

        tod.meta = {'latitude': self.site.latitude,
                    'longitude': self.site.longitude,
                    'altitude': self.site.altitude}

        return tod



