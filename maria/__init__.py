# AVE MARIA, gratia plena, Dominus tecum.
# Benedicta tu in mulieribus, et benedictus fructus ventris tui, Iesus.
# Sancta Maria, Mater Dei, ora pro nobis peccatoribus, nunc, et in hora mortis nostrae.

import logging

from . import utils  # noqa
from ._version import __version__, __version_tuple__  # noqa
from .instrument import Band, Instrument, all_instruments, get_instrument  # noqa
from .map import Map, mappers  # noqa
from .plan import Plan, all_plans, get_plan  # noqa
from .sim import Simulation  # noqa
from .site import Site, all_regions, all_sites, get_site  # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("maria")
