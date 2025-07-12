import os
import re

import numpy as np
import pandas as pd
import yaml

from ..utils import deg_to_signed_dms, deg_to_signed_hms
from .prefixes import PREFIXES

here, this_filename = os.path.split(__file__)

with open(f"{here}/quantities.yml") as f:
    QUANTITIES = yaml.safe_load(f)

units_entries = {}
for q, q_config in QUANTITIES.items():
    for unit, unit_entry in q_config.pop("units").items():
        units_entries[unit] = {**unit_entry, "quantity": q, "quantity_default_unit": q_config["default_unit"]}
        units_entries[unit]["aliases"] = [unit, *units_entries[unit].get("aliases", [])]

UNITS = pd.DataFrame(units_entries).fillna("").T
UNITS["factor"] = UNITS["factor"].astype(float)

prefixes_phrase = r"|".join(PREFIXES.index)
base_units_phrase = r"|".join([alias for _, entry in UNITS.iterrows() for alias in entry.aliases]).replace("^", "\\^")
units_pattern = re.compile(rf"^(?P<prefix>({prefixes_phrase}))(?P<base_unit>{base_units_phrase})$")  # noqa


def parse_units(u):
    match = units_pattern.search(u)
    if match is None:
        raise ValueError(
            f"Invalid units '{u}'. Valid units are a combination of an SI prefix "
            f"(one of {prefixes_phrase}) and a base unit (one of {base_units_phrase}).",
        )

    units_dict = match.groupdict()
    prefix = PREFIXES.loc[units_dict["prefix"]]
    for _, entry in UNITS.iterrows():
        if units_dict["base_unit"] in entry["aliases"]:
            base_unit = entry
            break

    units_dict["units"] = f"{prefix.name}{base_unit.name}"
    units_dict.update(base_unit.to_dict())
    units_dict["long_name"] = f"{prefix.long_name}{base_unit.long_name}"
    units_dict["math_name"] = f"{prefix.symbol_latex}{base_unit.math_name}"
    units_dict["factor"] = base_unit.factor * prefix.factor
    if not units_dict["symbol"]:
        units_dict.pop("symbol")
    return units_dict


class Quantity:
    """
    A number (or numbers) with units representing a quantity with some dimensions.
    """

    def __init__(self, value, units, force_units=False):
        self.u = parse_units(units)

        if isinstance(value, Quantity):
            value = value.to(units)
        if np.ndim(value) == 1:
            value = np.array([x.to(units) if isinstance(x, Quantity) else x for x in value])
        try:
            self.value = np.array(value).astype(float)
        except Exception:
            raise TypeError("'value' must be a number")

        if not force_units:
            self.humanize()

    def humanize(self):
        x = self.value * self.u["factor"]

        self.q = QUANTITIES[self.u["quantity"]]
        natural_units = UNITS.loc[(UNITS.quantity == self.u["quantity"]) & (UNITS.natural)].sort_values("factor")

        abs_x = np.abs(x)
        abs_fin_x = np.where((abs_x > 0) & np.isfinite(abs_x), abs_x, np.nan)

        if (abs_fin_x > 0).any():
            fid_x = 2 * np.nanquantile(abs_fin_x, q=0.99)
            unit_index = np.digitize(fid_x, [0, *natural_units.factor.values[1:], np.inf]) - 1
            unit = natural_units.iloc[unit_index]

            power = 3 * (np.log10(fid_x / unit.factor) // 3)
            power = np.clip(power, a_min=self.min_prefix_power, a_max=self.max_prefix_power)
            prefix = PREFIXES.set_index("power").loc[power].symbol
            self.u = parse_units(f"{prefix}{unit.name}")

        else:
            self.u = parse_units(self.u["quantity_default_unit"])

        self.value = x / self.u["factor"]

    def pin(self, units):
        return Quantity(value=self.to(units), units=units, force_units=True)

    @property
    def quantity(self) -> str:
        return self.u["quantity"]

    @property
    def units(self) -> str:
        return self.u["units"]

    @property
    def min_prefix_power(self) -> bool:
        return self.q.get("min_prefix_power", -30)

    @property
    def max_prefix_power(self) -> bool:
        return self.q.get("max_prefix_power", 30)

    @property
    def dms(self) -> str:
        if self.quantity != "angle":
            raise ValueError("'dms' is only for angles")
        sign, d, m, s = deg_to_signed_dms(self.deg)
        return f"{int(sign * d):>02}°{int(m):>02}’{s:.02f}”"

    @property
    def hms(self) -> str:
        if self.quantity != "angle":
            raise ValueError("'hms' is only for angles")
        sign, h, m, s = deg_to_signed_hms(self.deg)
        return f"{int(sign * h):>02}ʰ{int(m):>02}ᵐ{s:.02f}ˢ"

    def __repr__(self) -> str:
        u = self.u
        if (u["prefix"] == "") and ("symbol" in u):
            units_string = u["symbol"]
        else:
            units_string = f" {self.units}"
        value_repr = f"{self.value:.04g}" if np.isscalar(self.value) else self.value
        return f"{value_repr}{units_string}"

    def to(self, units) -> float:
        u = parse_units(units)
        if u["quantity"] != self.u["quantity"]:
            raise ValueError(f"Cannot convert quantity '{self.u['quantity']}' to quantity '{u['quantity']}'.")
        return self.value * self.u["factor"] / u["factor"]

    def mean(self):
        return Quantity(np.mean(self.value), units=self.units)

    def min(self):
        return Quantity(np.min(self.value), units=self.units)

    def max(self):
        return Quantity(np.max(self.value), units=self.units)

    @property
    def shape(self):
        return np.shape(self.value)

    @property
    def size(self):
        return np.size(self.value)

    @property
    def ndim(self):
        return np.ndim(self.value)

    def __getattr__(self, attr):
        try:
            return self.to(attr)
        except Exception:
            pass
        raise AttributeError(f"'Quantity' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return Quantity(self.value[key], units=self.units)

    def __format__(self, *args, **kwargs):
        return repr(self).__format__(*args, **kwargs)

    def __round__(self, ndigits=0):
        return Quantity(self.value.round(), self.currency)

    def __float__(self):
        return float(self.value)

    def __len__(self):
        return len(self.value)

    def __array__(self, copy=True):
        return np.array(self.value)

    def __neg__(self):
        return Quantity(-self.value, units=self.units)

    def convert_other(self, other):
        if isinstance(other, Quantity):
            if self.quantity == other.quantity:
                return other.to(self.units)
            else:
                raise TypeError(f"Cannot add quantity '{self.q['long_name']}' to quantity '{other.q['long_name']}'")
        elif np.all(other == 0):
            return Quantity(np.zeros(np.shape(other)), self.units)
        raise TypeError(f"{self} and {other} are incompatible quantities")

    def __eq__(self, other):
        return self.value == self.convert_other(other)

    def __lt__(self, other):
        return self.value < self.convert_other(other)

    def __gt__(self, other):
        return self.value > self.convert_other(other)

    def __le__(self, other):
        return self.value <= self.convert_other(other)

    def __ge__(self, other):
        return self.value >= self.convert_other(other)

    def __add__(self, other):
        addend = self.convert_other(other)
        return Quantity(self.value + addend, units=self.units)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Quantity):
            raise TypeError("Multiplying quantities is not supported.")
        return Quantity(self.value * other, units=self.units)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            if self.quantity == other.quantity:
                return self.value / other.to(self.units)
        else:
            return Quantity(self.value / other, units=self.units)

    def __rtruediv__(self, other):
        return self.convert_other(other) / self.value
