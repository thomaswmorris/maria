import logging
import os
import re
from functools import cached_property

import numpy as np
import pandas as pd
import yaml

from ..utils import deg_to_signed_dms, deg_to_signed_hms
from .prefixes import PREFIXES

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

PREFIXES = pd.read_csv(f"{here}/prefixes.csv", index_col=0)
PREFIXES.loc[""] = "", "", "", 0, 1e0
PREFIXES.sort_values("factor", ascending=True, inplace=True)
PREFIXES.loc[:, "primary"] = np.log10(PREFIXES.factor.values) % 3 == 0

dim_entries = {}
quantity_entries = {}
with open(f"{here}/quantities.yml") as f:
    unit_entries = {}
    for quantity, q in yaml.safe_load(f).items():
        q_units = q.pop("units")
        for unit, config in q_units.items():
            config["quantity"] = quantity
            config["aliases"] = set([unit, config["long_name"], *config.get("aliases", [])])
        unit_entries.update(q_units)
        dim_entries[quantity] = q["recipe"]
        quantity_entries[quantity] = q

QUANTITIES = pd.DataFrame(quantity_entries).T
DIMENSIONS = pd.DataFrame(dim_entries).sort_index().T.fillna(0)
UNITS = pd.DataFrame(unit_entries).T
UNITS["human"] = UNITS["human"].astype(bool).fillna(True)
UNITS["symbol"] = UNITS["symbol"].fillna("")
UNITS["factor"] = UNITS["factor"].astype(float)

prefixes_phrase = r"|".join(PREFIXES.index)
base_units_phrase = r"|".join([alias for _, entry in UNITS.iterrows() for alias in entry.aliases]).replace("^", "\\^")
units_pattern = re.compile(rf"^(?P<prefix>({prefixes_phrase}))(?P<raw_unit>{base_units_phrase})(\^(?P<power>.*))?$")  # noqa


def parse_units(units):
    match = units_pattern.search(units)
    if match is None:
        raise ValueError(
            f"Invalid units '{units}'. Valid units are a combination of an SI prefix "
            f"(one of {prefixes_phrase}) and a base unit (one of {base_units_phrase}).",
        )

    u = match.groupdict()
    for _, entry in UNITS.iterrows():
        if u["raw_unit"] in entry.aliases:
            u["raw_unit"] = entry.name
            break

    u.update(UNITS.loc[u["raw_unit"]])
    prefix = PREFIXES.loc[u["prefix"]]
    u["units"] = f"{prefix.name}{u['raw_unit']}"
    u["long_name"] = f"{prefix.long_name}{u['long_name']}"
    u["math_name"] = f"{prefix.symbol_latex}{u['math_name']}"
    u["factor"] *= prefix.factor
    if not u["symbol"]:
        u.pop("symbol")
    return u


def get_factor_and_base_units_vector(units):
    factor = 1
    base_units_vector = np.zeros(len(DIMENSIONS.columns))

    for subunit in units.split(" "):
        u = parse_units(subunit)
        u.update(UNITS.loc[u["raw_unit"]])
        u["power"] = float(u["power"]) if u["power"] else 1.0
        base_units_vector += u["power"] * np.array(DIMENSIONS.loc[u["quantity"]])
        factor *= (u["factor"] * PREFIXES.loc[u["prefix"], "factor"]) ** u["power"]

    return factor, pd.Series(base_units_vector, index=DIMENSIONS.columns)


def repr_power(thing, power):
    if power == 0:
        return None
    if power == 1:
        pow_string = ""
    elif power == int(power):
        pow_string = f"^{int(power)}"
    else:
        pow_string = f"^{power}"
    return f"{thing}{pow_string}"


class Quantity:
    def __new__(cls, value: float, units: str | pd.Series):
        self = super().__new__(cls)

        if isinstance(value, Quantity):
            self.base_units_value = value.base_units_value
            self.base_units_vector = value.base_units_vector

        else:
            if isinstance(units, pd.Series):
                self.base_units_value = np.array(value).astype(float)
                self.base_units_vector = units

            else:
                factor, base_units_vector = get_factor_and_base_units_vector(units)
                if np.ndim(value) == 1:
                    unique_quantities = np.unique([x.quantity for x in value if isinstance(x, Quantity)])
                    if len(unique_quantities) > 1:
                        raise ValueError("'value' contains Quantities of more than one dimension vector")
                    self.base_units_value = factor * np.array([x.to(units) if isinstance(x, Quantity) else x for x in value])
                    self.base_units_vector = base_units_vector

                elif isinstance(units, str):
                    self.base_units_value = factor * np.array(value).astype(float)
                    self.base_units_vector = base_units_vector

                else:
                    raise ValueError("'units' must be a string")

        if (self.base_units_vector == 0).all():
            return self.base_units_value

        return self

    @cached_property
    def quantity(self):
        return self.u["quantity"]

    @cached_property
    def q(self):
        return QUANTITIES.loc[self.quantity].to_dict()

    @cached_property
    def u(self):
        if not self.composite():
            return parse_units(self.units)
        return {
            "prefix": "",
            "raw_units": self.base_units,
            "power": None,
            "factor": 1e0,
            "long_name": "jansky",
            "math_name": f"\\text{{{self.base_units}}}",
            "min_prefix_power": 0,
            "max_prefix_power": 0,
            "human": True,
            "quantity": self.base_units,
            "aliases": {},
            "symbol": "",
            "units": self.base_units,
        }

    def to(self, units):
        factor, base_units_vector = get_factor_and_base_units_vector(units)
        if (base_units_vector == self.base_units_vector).all():
            return self.base_units_value / factor
        else:
            raise ValueError()

    def composite(self):
        quantity_units = self.known_compatible_units()
        return len(quantity_units) == 0

    @cached_property
    def quantity(self):
        qmask = (self.base_units_vector == DIMENSIONS).all(axis=1)
        if qmask.sum() > 0:
            return DIMENSIONS.loc[qmask].iloc[0].name
        else:
            return self.base_units

    def known_compatible_units(self):
        return UNITS.loc[UNITS.quantity == self.quantity]

    @cached_property
    def base_units(self):
        if not hasattr(self, "_base_units"):
            unit_parts = []
            for unit, power in self.base_units_vector.items():
                part = repr_power(unit, power)
                if part is not None:
                    unit_parts.append(part)
            self._base_units = " ".join(unit_parts)
        return self._base_units

    def pin(self, units, inplace=False):
        if inplace:
            self.pinned_units = units
        else:
            pinned_quantity = type(self)(self.base_units_value, self.base_units)
            pinned_quantity.pin(units, inplace=True)
            return pinned_quantity

    @cached_property
    def value(self):
        return self.to(self.units)

    @cached_property
    def units(self):
        if hasattr(self, "pinned_units"):
            return self.pinned_units

        abs_value = np.abs(self.base_units_value)
        abs_finite_value = np.where((abs_value > 0) & np.isfinite(abs_value), abs_value, np.nan)

        if (abs_finite_value > 0).any():
            fid_x = 2 * np.nanquantile(abs_finite_value, q=0.99)

            # all the named units that work for these dimensions
            raw_quantity_units = self.known_compatible_units()
            raw_quantity_units = raw_quantity_units.loc[raw_quantity_units.human]

            if len(raw_quantity_units) > 0:
                prefixed_quantity_units = pd.DataFrame(columns=["value"])
                for unit, unit_entry in raw_quantity_units.iterrows():
                    prefix_mask = (
                        (PREFIXES.power % 3 == 0)
                        & (unit_entry.min_prefix_power <= PREFIXES.power)
                        & (PREFIXES.power <= unit_entry.max_prefix_power)
                    )
                    for prefix, prefix_entry in PREFIXES.loc[prefix_mask].iterrows():
                        prefixed_quantity_units.loc[f"{prefix}{unit}"] = fid_x / (prefix_entry.factor * unit_entry.factor)

                ideal_mask = (prefixed_quantity_units.value > 0.5) & (prefixed_quantity_units.value < 500)
                if ideal_mask.sum() > 0:
                    repr_units = prefixed_quantity_units.loc[ideal_mask].sort_values("value").iloc[0].name
                else:
                    score = np.abs(np.log(prefixed_quantity_units.value / 16))
                    repr_units = prefixed_quantity_units.iloc[score.argmin()].name
                return repr_units

        return self.base_units

    def __repr__(self) -> str:
        if not hasattr(self, "value"):
            if not self.composite():
                self.units = self.compute_human_units()
                self.value = self.to(self.units)
            else:
                self.units = self.base_units
                self.value = self.base_units_value

        if self.quantity == "time":
            if self.s > 3600:
                return self.timestring
        value_repr = f"{self.value:.04g}" if np.isscalar(self.value) else self.value
        return f"{value_repr}{UNITS['symbol'].get(self.units) or f' {self.units}'}"

    def __neg__(self):
        return type(self)(-self.base_units_value, units=self.base_units_vector)

    def __add__(self, other):
        if (self.base_units_vector == other.base_units_vector).all():
            return type(self)(self.base_units_value + other.base_units_value, units=self.base_units_vector)
        else:
            raise RuntimeError(f"Cannot add units {self} and {other}")

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return type(self)(
                self.base_units_value * other.base_units_value, units=self.base_units_vector + other.base_units_vector
            )
        else:
            return type(self)(self.base_units_value * other, units=self.base_units_vector)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return type(self)(
                self.base_units_value / other.base_units_value, units=self.base_units_vector - other.base_units_vector
            )
        else:
            return type(self)(self.base_units_value / other, units=self.base_units_vector)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __pow__(self, power):
        return type(self)(self.base_units_value**power, units=self.base_units_vector * power)

    def __copy__(self):
        return Quantity(self.base_units_value, self.base_units_vector)

    def __deepcopy__(self, memo):
        return self.__copy__()

    @property
    def timestring(self) -> str:
        if self.quantity != "time":
            raise ValueError("'timestring' is only for times")
        parts = []
        t = self.seconds
        for k, v in {"y": 365 * 86400, "d": 86400, "h": 3600, "m": 60}.items():
            if t > v:
                parts.append(f"{int(t // v)}{k}")
                t = t % v
        parts.append(f"{t:.03f}s")
        return " ".join(parts)

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

    def mean(self, axis=None, *args, **kwargs):
        return Quantity(np.mean(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units)

    def median(self, axis=None, *args, **kwargs):
        return Quantity(np.median(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units)

    def min(self, axis=None, *args, **kwargs):
        return Quantity(np.min(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units)

    def max(self, axis=None, *args, **kwargs):
        return Quantity(np.max(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units)

    def std(self, axis=None, *args, **kwargs):
        return Quantity(np.std(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units)

    def ptp(self, axis=None, *args, **kwargs):
        return Quantity(np.ptp(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units)

    @property
    def shape(self):
        return np.shape(self.base_units_value)

    @property
    def size(self):
        return np.size(self.base_units_value)

    @property
    def ndim(self):
        return np.ndim(self.base_units_value)

    def __getattr__(self, attr):
        try:
            return self.to(attr)
        except Exception:
            pass
        raise AttributeError(f"'Quantity' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return Quantity(self.base_units_value[key], units=self.base_units)

    def __format__(self, *args, **kwargs):
        return repr(self).__format__(*args, **kwargs)

    # def __float__(self):
    #     return float(self.base_units_value)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return len(self.base_units_value)

    def __neg__(self):
        return Quantity(-self.base_units_value, units=self.base_units)

    def convert_other(self, other):
        if isinstance(other, Quantity):
            if (self.base_units_vector == other.base_units_vector).all():
                return other.to(self.base_units)
            else:
                raise TypeError(f"Cannot combine quantities '{self.quantity}' to quantity '{other.quantity}'")
        elif np.all(other == 0):
            return other
        raise TypeError(f"{self} and {other} are incompatible quantities")

    def __eq__(self, other):
        return self.base_units_value == self.convert_other(other)

    def __lt__(self, other):
        return self.base_units_value < self.convert_other(other)

    def __gt__(self, other):
        return self.base_units_value > self.convert_other(other)

    def __le__(self, other):
        return self.base_units_value <= self.convert_other(other)

    def __ge__(self, other):
        return self.base_units_value >= self.convert_other(other)
