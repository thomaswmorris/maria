from __future__ import annotations

from .constants import MARIA_MAX_NU_HZ, MARIA_MIN_NU_HZ
from .units import Quantity


class PointingError(Exception): ...


class ShapeError(Exception): ...


class ConfigurationError(Exception): ...


class FrequencyOutOfBoundsError(Exception):
    def __init__(self, nu=None, center_and_width=None):
        qmin_nu = Quantity(MARIA_MIN_NU_HZ, units="Hz")
        qmax_nu = Quantity(MARIA_MAX_NU_HZ, units="Hz")

        if nu:
            super().__init__(
                f"Bad frequencies nu={Quantity(nu, 'Hz')}; maria supports frequencies between {qmin_nu} and {qmax_nu}."
            )

        if center_and_width:
            center, width = center_and_width
            super().__init__(
                f"Band with center {Quantity(center, 'Hz')} and width {Quantity(width, 'Hz')} has a passband extending "
                f"out of bounds; maria supports frequencies between {qmin_nu} and {qmax_nu}."
            )
