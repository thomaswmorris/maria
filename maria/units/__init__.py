from __future__ import annotations

import os

import astropy.constants as const
import numpy as np
from astropy import units as u

from ..constants import T_CMB, c, g, k_B  # noqa
from .angle import Angle  # noqa
from .prefixes import PREFIXES  # noqa
from .quantity import QUANTITY_DIMENSION_VECTORS, UNITS, Quantity, parse_units  # noqa

here, this_filename = os.path.split(__file__)
symbols = {"radians": "rad"}

# for unit in UNITS.index:
#     try:
#         exec(f"{unit} = Quantity(1, '{unit}')")
#     except SyntaxError:
#         continue
