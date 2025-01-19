import time as ttime
import logging
from ..units import prefixes


def humanize(x, unit=""):
    for _, prefix in prefixes.iterrows():
        if prefix.primary:
            value = x / prefix.factor
            if value < 1e3:
                break
    if value > 100:
        value = round(value)
    elif value > 10:
        value = round(value, 1)
    else:
        value = round(value, 2)
    return f"{value} {prefix.name}{unit}"


def humanize_time(seconds):
    return humanize(seconds, unit="s")


def log_duration(ref_time, message, level="debug"):
    logger = logging.getLogger("maria")
    string = f"{message} in {humanize_time(ttime.monotonic() - ref_time)}."
    getattr(logger, level)(string)
