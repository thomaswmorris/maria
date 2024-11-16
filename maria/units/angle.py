from __future__ import annotations

import numpy as np

UNITS = {
    "radians": {"short_name": "rad", "factor": 1.0, "symbol": "rad"},
    "degrees": {"short_name": "deg", "factor": 180 / np.pi, "symbol": "°"},
    "arcminutes": {"short_name": "arcmin", "factor": 60 * 180 / np.pi, "symbol": "’"},
    "arcseconds": {"short_name": "arcsec", "factor": 3600 * 180 / np.pi, "symbol": "”"},
}


class Angle:
    def __init__(self, a, units="radians"):
        self.radians = None
        for k in UNITS:
            if units in [k, UNITS[k]["short_name"]]:
                self.radians = a / UNITS[k]["factor"]
        if self.radians is None:
            raise ValueError(f"Invalid units '{units}'.")

        self.is_scalar = len(np.shape(self.radians)) == 0

        if not self.is_scalar:
            self.radians = np.unwrap(self.radians)

    def __getattr__(self, attr):
        for k in UNITS:
            if attr in [k, UNITS[k]["short_name"]]:
                return self.radians * UNITS[k]["factor"]
        raise ValueError(f"Angle object has no attribute named '{attr}'.")

    def __float__(self):
        return self.rad

    def __repr__(self):
        units = self.units
        if self.is_scalar:
            return f"{getattr(self, units)}{UNITS[units]['symbol']}"
        else:
            return f"Angle({getattr(self, units)}, units={units})"

    @property
    def units(self):
        # peak-to-peak
        max_deg = self.deg if self.is_scalar else self.deg.max()
        if max_deg < 0.5 / 60:
            return "arcseconds"
        if max_deg < 0.5:
            return "arcminutes"
        return "degrees"

    @property
    def units_short(self):
        return UNITS[self.units]["short_name"]

    @property
    def values(self):
        return getattr(self, self.units)
