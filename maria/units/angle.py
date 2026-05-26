from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("maria")

UNITS = {
    "radians": {"short_name": "rad", "factor": 1.0, "symbol": "rad"},
    "degrees": {"short_name": "deg", "factor": 180 / np.pi, "symbol": "°"},
    "arcminutes": {"short_name": "arcmin", "factor": 60 * 180 / np.pi, "symbol": "’"},
    "arcseconds": {"short_name": "arcsec", "factor": 3600 * 180 / np.pi, "symbol": "”"},
}


class Angle:
    def __init__(self, a, units: str = "radians", unwrap: bool = False):
        self.x = None
        self.unwrap = unwrap
        for k in UNITS:
            if units in [k, UNITS[k]["short_name"]]:
                self.x = np.array(a) / UNITS[k]["factor"]
        if self.x is None:
            raise ValueError(f"Invalid units '{units}'.")

        self.is_scalar = len(np.shape(self.radians)) == 0

        if self.is_scalar and unwrap:
            raise ValueError()

        # 3 April 2025
        logger.warning(
            "The Angle class is deprecated and will be removed in a future update. "
            "Please refactor your code using the Quantity class (maria.units.Quantity)."
        )

    def __getattr__(self, attr):
        radians = self.x if self.unwrap else (self.x + np.pi) % (2 * np.pi) - np.pi
        for k in UNITS:
            if attr in [k, UNITS[k]["short_name"]]:
                return radians * UNITS[k]["factor"]
        raise AttributeError(f"'Angle' object has no attribute '{attr}'.")

    def __getitem__(self, idx):
        return self.radians[idx]

    def __float__(self):
        return self.radians

    def __repr__(self):
        units = self.units
        if self.is_scalar:
            return f"{round(getattr(self, units), 3)}{self.symbol}"
        else:
            return f"Angle({getattr(self, units)}, units={units})"

    @property
    def shape(self):
        return self.radians.shape

    @property
    def symbol(self):
        return UNITS[self.units]["symbol"]

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
