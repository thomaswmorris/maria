# AVE MARIA, gratia plena, Dominus tecum.
# Benedicta tu in mulieribus, et benedictus fructus ventris tui, Iesus.
# Sancta Maria, Mater Dei, ora pro nobis peccatoribus, nunc, et in hora mortis nostrae.

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import array, pointing, site, utils, weather  # noqa F401
from .array import get_array  # noqa F401
from .pointing import get_pointing  # noqa F401
from .sim import Simulation  # noqa F401
from .site import get_site  # noqa F401
