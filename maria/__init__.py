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
import copy

import weathergen
from tqdm import tqdm

import warnings
import healpy as hp

from matplotlib import pyplot as plt
from astropy.io import fits

from .array import get_array, get_array_config
from .pointing import get_pointing, get_pointing_config
from .site import get_site, get_site_config

from .sim import Simulation

here, this_filename = os.path.split(__file__)