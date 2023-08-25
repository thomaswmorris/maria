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



from . import utils
from .tod import TOD

from astropy.io import fits

here, this_filename = os.path.split(__file__)

class BaseSimulation:
    """
    The base class for a simulation. This is an ingredient in every simulation.
    """

    def __init__(self, 
                 array, 
                 pointing, 
                 site):

        self.array, self.pointing, self.site = array, pointing, site
        self.coordinator = utils.Coordinator(lat=self.site.latitude, lon=self.site.longitude)

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



