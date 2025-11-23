from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from typing import Mapping

import arrow
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..coords import FRAMES, Frame, infer_center_width_height
from ..instrument import BandList
from ..io import DEFAULT_BAR_FORMAT, repr_phi_theta
from ..map import MAP_QUANTITIES, ProjectionMap
from ..tod import TOD, TOD_QUANTITIES
from ..units import Quantity, parse_units
from .base import BaseMapper

# np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")
