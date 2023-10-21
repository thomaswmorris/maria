import os
import time as ttime

import h5py
import numpy as np
import scipy as sp
from tqdm import tqdm

from .. import base, utils, weather

# how do we do the bands? this is a great question.
# because all practical telescope instrumentation assume a constant band

from .single_layer import SingleLayerSimulation
# from .kolmogorov_taylor import KolmogorovTaylorSimulation
# from .linear_angular import LinearAngularSimulation

ATMOSPHERE_PARAMS = {
    "min_depth": 500,
    "max_depth": 3000,
    "n_layers": 4,
    "min_beam_res": 4,
}




