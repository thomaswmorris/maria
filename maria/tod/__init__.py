from . import signal, tod  # noqa
from .processing import process_tod
from .tod import TOD  # noqa

TOD.process = process_tod  # to avoid circular imports
