import numpy as np

from ..units import Quantity


def humanize(x, units):
    return str(Quantity(x, units=units))


def humanize_time(seconds):
    return humanize(seconds, units="s")


def leftpad(thing, n: int = 2, char=" "):
    return "\n".join([n * char + line for line in str(thing).splitlines()])
