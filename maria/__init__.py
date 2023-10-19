# Ave, María, grátia plena, Dóminus tecum
from ._version import get_versions
__version__ = get_versions()["version"]

del get_versions

from . import array, pointing, site, utils, weather

from .array import get_array, get_array_config
from .pointing import get_pointing, get_pointing_config
from .site import get_site, get_site_config
from .sim import Simulation