import logging
import time as ttime

import numpy as np

from ..units import Quantity


def humanize(x, units):
    return str(Quantity(x, units=units))


def humanize_time(seconds):
    return humanize(seconds, units="s")


def log_duration(ref_time, message, level="debug"):
    logger = logging.getLogger("maria")
    string = f"{message} in {humanize_time(ttime.monotonic() - ref_time)}."
    getattr(logger, level)(string)
