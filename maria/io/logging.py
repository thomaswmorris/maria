import logging
import time as ttime

from .repr import humanize_time

DEFAULT_TIME_FORMAT = "YYYY-MM-DD HH:mm:ss.SSS ZZ"
DEFAULT_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"


def log_duration(ref_time, message, level="debug"):
    logger = logging.getLogger("maria")
    string = f"{message} in {humanize_time(ttime.monotonic() - ref_time)}."
    getattr(logger, level)(string)
