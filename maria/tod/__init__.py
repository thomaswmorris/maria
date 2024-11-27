from __future__ import annotations

from . import signal, tod  # noqa
from .processing import process_tod  # noqa
from .tod import TOD  # noqa
from .field import Field  # noqa

TOD.process = process_tod  # to avoid circular imports
