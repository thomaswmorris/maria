from __future__ import annotations

import logging
import os

from ..spectrum import AtmosphericSpectrum
from .atmosphere import Atmosphere

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

SUPPORTED_MODELS_LIST = ["2d", "3d"]

DEFAULT_ATMOSPHERE_KWARGS = {}
