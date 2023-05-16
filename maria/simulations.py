
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

here, this_filename = os.path.split(__file__)

from . import utils

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


        

        

