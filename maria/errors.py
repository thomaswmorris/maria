from __future__ import annotations

from .constants import MARIA_MAX_NU, MARIA_MIN_NU
from .units import Quantity


class PointingError(BaseException):
    pass


class ConfigurationError(Exception):
    pass


class FrequencyOutOfBoundsError(Exception):
    def __init__(self, freqs):
        qmin_nu = Quantity(MARIA_MIN_NU, units="Hz")
        qmax_nu = Quantity(MARIA_MAX_NU, units="Hz")
        super().__init__(
            f"Bad frequencies nu={Quantity(freqs, 'Hz')}; maria supports frequencies between {qmin_nu} and {qmax_nu}."
        )
