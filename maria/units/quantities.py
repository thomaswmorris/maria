import os
import re

import numpy as np
import pandas as pd
import yaml

from .prefixes import PREFIXES

here, this_filename = os.path.split(__file__)

with open(f"{here}/quantities.yml") as f:
    QUANTITIES = yaml.safe_load(f)

units_entries = {}
for q, q_config in QUANTITIES.items():
    for unit, unit_entry in q_config.pop("units").items():
        units_entries[unit] = {**unit_entry, "quantity": q, "default_units": q_config["default_units"]}

UNITS = pd.DataFrame(units_entries).T
UNITS["factor"] = UNITS["factor"].astype(float)

prefixes_phrase = "|".join(PREFIXES.index)
base_units_phrase = "|".join(UNITS.index.values)
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
    base_unit = UNITS.loc[units_dict["base_unit"]]
    units_dict["units"] = f"{prefix.name}{base_unit.name}"
    units_dict.update(base_unit.to_dict())
    units_dict["long_name"] = f"{prefix.long_name}{base_unit.long_name}"
    units_dict["math_name"] = f"{prefix.symbol_latex}{base_unit.math_name}"
    units_dict["factor"] = base_unit.factor * prefix.factor
    return units_dict


class Quantity:
    """
    A number (or numbers) with units representing a quantity with some dimensions.
    """

    def __init__(self, value, units):
        u = parse_units(units)
        x = np.array(value) * u["factor"]  # the

        # u = parse_units(u["default_unit"]) # the

        self.q = QUANTITIES[u["quantity"]]
        natural_units = UNITS.loc[(UNITS.quantity == u["quantity"]) & (UNITS.natural)].sort_values("factor")

        abs_x = np.abs(x)

        if (abs_x > 0).any():
            fid_x = np.nanquantile(np.where(abs_x > 0, abs_x, np.nan), q=0.95)
            unit_index = np.digitize(fid_x, [0, *natural_units.factor.values[1:], np.inf]) - 1
            unit = natural_units.iloc[unit_index]

            power = 3 * (np.log10(fid_x / unit.factor) // 3)
            power = np.clip(power, a_min=self.min_prefix_power, a_max=self.max_prefix_power)
            prefix = PREFIXES.set_index("power").loc[power].symbol
            self.u = parse_units(f"{prefix}{unit.name}")

        else:
            self.u = parse_units(u["default_units"])

        self.value = x / self.u["factor"]

    @property
    def units(self) -> str:
        return self.u["units"]

    # @property
    # def human_units_math(self) -> str:
    #     return parse_units(self.human_units)["math_name"]

    @property
    def min_prefix_power(self) -> bool:
        return self.q.get("min_prefix_power", -30)

    @property
    def max_prefix_power(self) -> bool:
        return self.q.get("max_prefix_power", 30)

    def __repr__(self) -> str:
        value_repr = f"{self.value:.04g}" if np.isscalar(self.value) else self.value
        return f"{value_repr} {self.units}"

    def to(self, units) -> float:
        u = parse_units(units)
        if u["quantity"] != self.u["quantity"]:
            raise ValueError(f"Cannot convert quantity '{self.u['quantity']}' to quantity '{u['quantity']}'.")
        return self.value * self.u["factor"] / u["factor"]

    def __getattr__(self, attr):
        try:
            return self.to(attr)
        except Exception:
            pass
        raise AttributeError(f"'Quantity' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return Quantity(self.value[key], units=self.units)

    def mean(self):
        return Quantity(np.mean(self.value), units=self.u["units"])

    def __add__(self, other):
        return Quantity(self.value + getattr(other, self.units), units=self.units)

    def __sub__(self, other):
        return Quantity(self.value - getattr(other, self.units), units=self.units)
