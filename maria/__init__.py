# AVE MARIA, gratia plena, Dominus tecum.
# Benedicta tu in mulieribus, et benedictus fructus ventris tui, Iesus.
# Sancta Maria, Mater Dei, ora pro nobis peccatoribus, nunc, et in hora mortis nostrae.

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import array, pointing, site, utils, weather  # noqa F401
from .array import all_arrays, get_array  # noqa F401
from .pointing import all_pointings, get_pointing  # noqa F401
from .sim import Simulation  # noqa F401
from .site import all_sites, get_site  # noqa F401
from .tod import TOD  # noqa F401
from .weather import all_regions  # noqa F401
