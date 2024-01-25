# AVE MARIA, gratia plena, Dominus tecum.
# Benedicta tu in mulieribus, et benedictus fructus ventris tui, Iesus.
# Sancta Maria, Mater Dei, ora pro nobis peccatoribus, nunc, et in hora mortis nostrae.

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import utils  # noqa F401
from .array import all_arrays, get_array  # noqa F401
from .map import Map, mappers  # noqa F401
from .pointing import all_pointings, get_pointing  # noqa F401
from .sim import Simulation  # noqa F401
from .site import all_regions, all_sites, get_site  # noqa F401
from .tod import TOD  # noqa F401
