import numpy as np

from ..units import Quantity


def humanize(x, units):
    return str(Quantity(x, units=units))


def humanize_time(seconds):
    return humanize(seconds, units="s")


def leftpad(thing, n: int = 2, char=" "):
    return "\n".join([n * char + line for line in str(thing).splitlines()])


def latex_scientific_notation_repr(x):
    if not np.size(x) == 1:
        raise ValueError()
    power = np.floor(np.log10(x))
    return rf"{x * 10**-power:.2f} \times 10^{{{int(power)}}}"
