# AVE MARIA, gratia plena, Dominus tecum.
# Benedicta tu in mulieribus, et benedictus fructus ventris tui, Iesus.
# Sancta Maria, Mater Dei, ora pro nobis peccatoribus, nunc, et in hora mortis nostrae.
from __future__ import annotations

import logging

from ._version import __version__, __version_tuple__  # noqa
from .array import Array, all_arrays, get_array  # noqa
from .band import Band, all_bands, get_band  # noqa
from .instrument import Instrument, all_instruments, get_instrument  # noqa
from .io import fetch, set_cache_dir  # noqa
from .map import all_maps  # noqa
from .plan import Plan, all_plans, get_plan  # noqa
from .sim import Simulation  # noqa
from .site import Site, all_regions, all_sites, get_site  # noqa

logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("maria")


def debug():
    logger.setLevel(logging.DEBUG)
