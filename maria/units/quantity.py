import logging
import os
import pathlib
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

unit_entries = {}
quantity_entries = {}
dimension_vector_entries = {}


class InvalidUnitError(Exception): ...


for path in pathlib.Path(f"{here}/physical_quantities").glob("*.yml"):
    with open(path) as f:
        for quantity, q in yaml.safe_load(f).items():
            if "units" in q:
                q_units = q.pop("units")
                for unit, config in q_units.items():
                    config["physical_quantity"] = quantity
                    config["aliases"] = set([unit, config["long_name"], *config.get("aliases", [])])
                unit_entries.update(q_units)
            quantity_entries[quantity] = q
            dimension_vector_entries[quantity] = q["vector"]

UNITS = pd.DataFrame(unit_entries).T
QUANTITIES = pd.DataFrame(quantity_entries).T
QUANTITY_DIMENSION_VECTORS = pd.DataFrame(dimension_vector_entries).sort_index().T.fillna(0)

UNITS["units"] = UNITS.index
UNITS["human"] = UNITS["human"].astype(bool).fillna(True)
UNITS["symbol"] = UNITS["symbol"].fillna("")
UNITS["factor"] = UNITS["factor"].astype(float)

prefixes_phrase = r"|".join(PREFIXES.index)
base_units_phrase = r"|".join([alias for _, entry in UNITS.iterrows() for alias in entry.aliases]).replace("^", "\\^")
units_pattern = re.compile(rf"^(?P<prefix>({prefixes_phrase}))(?P<raw_unit>{base_units_phrase})((\^|\*\*)(?P<power>.*))?$")  # noqa


def repr_power(thing: str, power: float, math: bool = False):
    exp_numer, exp_denom = power.as_integer_ratio()

    if exp_numer % exp_denom:
        exp_string = f"{exp_numer}/{exp_denom}" if math else f"{power}"
    else:
        exp_string = f"{int(exp_numer / exp_denom)}"
    if math:
        exp_string = f"{{{exp_string}}}"

    if power == 0:
        return ""
    if power == 1:
        return thing

    return f"{thing}^{exp_string}"


def repr_dim_vec(dim_vec: pd.Series) -> str:
    base_unit_parts = []
    for unit, power in dim_vec.items():
        part = repr_power(unit, power)
        if part is not None:
            base_unit_parts.append(part)
    return " ".join(base_unit_parts)


def parse_units(units):
    factor = 1
    dimension_vector = pd.Series(0.0, index=QUANTITY_DIMENSION_VECTORS.columns, dtype=float)

    math_name_parts = []

    for subunit in re.compile(r"(/?[\w\*\^\-\.√]+)").findall(units):
        match = units_pattern.search(subunit.strip("/"))
        if match is None:
            raise InvalidUnitError(
                f"Invalid units '{subunit.strip('/')}'. Valid units are a combination of an SI prefix "
                f"(one of {prefixes_phrase}) and a base unit (one of {base_units_phrase}).",
            )

        u = match.groupdict()
        for _, entry in UNITS.iterrows():
            if u["raw_unit"] in entry.aliases:
                u["raw_unit"] = entry.name

        u.update(UNITS.loc[u["raw_unit"]])
        prefix = PREFIXES.loc[u["prefix"]]
        power = (float(u["power"]) if u["power"] else 1.0) * (-1 if subunit[0] == "/" else 1)

        dimension_vector += power * np.array(QUANTITY_DIMENSION_VECTORS.loc[u["physical_quantity"]])
        factor *= (u["factor"] * prefix.factor) ** power
        math_name_parts.append(repr_power(f"{prefix.math_name}{u['math_name']}", power, math=True))

    physical_quantity_match = (QUANTITY_DIMENSION_VECTORS == dimension_vector).all(axis=1)
    physical_quantity = (
        physical_quantity_match.loc[physical_quantity_match].index[0] if physical_quantity_match.any() else "composite"
    )

    return {
        "units": units,
        "math_name": r"\ ".join(math_name_parts),
        "base_units_factor": factor,
        "base_units": repr_dim_vec(dimension_vector),
        "physical_quantity": physical_quantity,
        "dimension_vector": dimension_vector,
    }


def lazy_nanquantile(x, q: float, laziness: int = 16, axis=None):
    return np.nanquantile(x.ravel()[::laziness], q=q, axis=axis)


class Quantity:
    """

    A number is a number, but some numbers have QUANTITIES (like length, time, etc.)

    A "quantity" is a number with a dimension, like 1
    A "unit" is an amount of some dimension.
    A "dimension" is an attribute of a unit.

    """

    def __new__(cls, value: float, units: str | pd.Series):
        self = super().__new__(cls)

        if isinstance(value, Quantity):
            units = value.units
            value = value.to(units)

        if isinstance(units, pd.Series):
            units = repr_dim_vec(units)

        for attr, attr_value in parse_units(units).items():
            setattr(self, attr, attr_value)

        self.u = parse_units(units)
        self.base_units = repr_dim_vec(self.u["dimension_vector"])

        if np.ndim(value) == 1:
            unique_base_units = np.unique([x.base_units for x in value if isinstance(x, Quantity)])
            if len(unique_base_units) > 1:
                raise ValueError("Cannot combine Quantity objects with different QUANTITIES.")
            self.base_units_value = self.u["base_units_factor"] * np.array(
                [x.to(units) if isinstance(x, Quantity) else x for x in value]
            )

        elif isinstance(units, str):
            self.base_units_value = self.u["base_units_factor"] * np.array(value).astype(float)

        else:
            raise ValueError("'units' must be a string")

        if (self.u["dimension_vector"] == 0).all():
            return self.base_units_value

        return self

    def humanize(self, verbose: bool = False):
        self._human_value = self.base_units_value
        self._human_units = self.base_units

        if self.u["physical_quantity"] != "composite":
            physical_quantity_units = UNITS.loc[UNITS.physical_quantity == self.u["physical_quantity"]]

            units_loss = np.inf
            fid_x = np.array([x for x in lazy_nanquantile(np.abs(self.base_units_value), q=[0.001, 0.5, 0.999]) if x > 0])

            if len(fid_x):
                for unit_name, unit in physical_quantity_units.iterrows():
                    if not unit.human:
                        continue

                    for prefix_name, prefix in PREFIXES.iterrows():
                        if prefix.power % 3 != 0:
                            continue

                        if prefix.power < unit.min_prefix_power:
                            continue

                        if prefix.power > unit.max_prefix_power:
                            continue

                        unit_value = fid_x / (unit.factor * prefix.factor)
                        loss = np.sum(np.where(unit_value >= 1, np.log10(unit_value), 3 + abs(np.log10(unit_value))))

                        if verbose:
                            print(prefix_name, unit_name, unit_value, loss)

                        if units_loss > loss:
                            total_factor = 1 / (unit.factor * prefix.factor)
                            units = f"{prefix_name}{unit_name}"
                            units_loss = loss

                self._human_value = total_factor * self.base_units_value
                self._human_units = units

    def to(self, units) -> float:
        u = parse_units(units)
        if (u["dimension_vector"] == self.dimension_vector).all():
            return self.base_units_value / u["base_units_factor"]
        else:
            raise ValueError()

    def pin(self, units, inplace=False):
        if inplace:
            self.pinned_units = units
        else:
            pinned_quantity = type(self)(self.base_units_value, self.base_units)
            pinned_quantity.pin(units, inplace=True)
            return pinned_quantity

    def __repr__(self) -> str:
        # if not hasattr(self, "value"):
        #     if not self.composite():
        #         self.units = self.compute_human_units()
        #         self.value = self.to(self.units)
        #     else:
        #         self.units = self.base_units
        #         self.value = self.base_units_value

        if self.physical_quantity == "time":
            if np.any(self.s > 3600):
                return self.timestring
        value_repr = f"{self.human_value:.04g}" if np.isscalar(self.human_value) else self.human_value
        return f"{value_repr}{UNITS['symbol'].get(self.human_units) or f' {self.human_units}'}"

    def __neg__(self):
        return type(self)(-self.base_units_value, units=self.dimension_vector)

    def __add__(self, other):
        if (self.dimension_vector == other.dimension_vector).all():
            return type(self)(self.base_units_value + other.base_units_value, units=self.dimension_vector)
        else:
            raise RuntimeError(f"Cannot add units {self} and {other}")

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return type(self)(
                self.base_units_value * other.base_units_value, units=self.dimension_vector + other.dimension_vector
            )
        else:
            return type(self)(self.base_units_value * other, units=self.dimension_vector)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return type(self)(
                self.base_units_value / other.base_units_value, units=self.dimension_vector - other.dimension_vector
            )
        else:
            return type(self)(self.base_units_value / other, units=self.dimension_vector)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __pow__(self, power):
        return type(self)(self.base_units_value**power, units=self.dimension_vector * power)

    # def __iter__(self):
    #     u = self.units
    #     for x in range(self.value):
    #         yield Quantity(x, units=u)

    def __copy__(self):
        return Quantity(self.base_units_value, self.dimension_vector)

    def __deepcopy__(self, memo):
        return self.__copy__()

    @property
    def timestring(self) -> str:
        if self.physical_quantity != "time":
            raise ValueError("'timestring' is only for times")
        parts = []
        t = self.seconds
        for k, v in {"y": 365 * 86400, "d": 86400, "h": 3600, "m": 60}.items():
            if t > v:
                parts.append(f"{int(t // v)}{k}")
                t = t % v
        parts.append(f"{t:.03f}s")
        return " ".join(parts)

    def repr(self, format: str) -> str:
        if format == "dms":
            if self.physical_quantity != "angle":
                raise ValueError("string format 'dms' is only for angles")
            sign, d, m, s = deg_to_signed_dms(self.deg)
            return f"{int(sign * d):>02}°{int(m):>02}’{s:.02f}”"

        if format == "hms":
            if self.physical_quantity != "angle":
                raise ValueError("string format 'hms' is only for angles")
            sign, h, m, s = deg_to_signed_hms(self.deg)
            return f"{int(sign * h):>02}ʰ{int(m):>02}ᵐ{s:.02f}ˢ"

        if format == "deg":
            if self.physical_quantity != "angle":
                raise ValueError("string format 'deg' is only for angles")
            return f"{self.deg:.04f}°"

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
        if attr in ["human_value", "human_units", "hu"]:
            if not hasattr(self, f"_{attr}"):
                self.humanize()
                self._hu = parse_units(self._human_units)
            return getattr(self, f"_{attr}")
        try:
            return self.to(attr)
        except Exception:
            pass
        raise AttributeError(f"'Quantity' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return Quantity(self.base_units_value[key], units=self.base_units)

    def __format__(self, *args, **kwargs):
        return repr(self).__format__(*args, **kwargs)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return len(self.base_units_value)

    def __neg__(self):
        return Quantity(-self.base_units_value, units=self.base_units)

    def convert_other(self, other):
        if isinstance(other, Quantity):
            if (self.dimension_vector == other.dimension_vector).all():
                return other.to(self.base_units)
            else:
                raise TypeError(
                    f"Cannot combine quantities '{self.physical_quantity}' to quantity '{other.physical_quantity}'"
                )
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
