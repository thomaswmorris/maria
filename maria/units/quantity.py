import logging
import os

import arrow
import numpy as np
import pandas as pd

from ..utils import compute_resolution_precision, deg_to_signed_dms, deg_to_signed_hms
from .prefixes import PREFIXES
from .units import UNITS, UnitError, parse_units, repr_dim_vec

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


def lazy_nanquantile(x, q: float, laziness: int = 16, axis=None):
    return np.nanquantile(x.ravel()[::laziness], q=q, axis=axis)


class Quantity:
    """

    A number is a number, but some numbers have QUANTITIES (like length, time, etc.)

    A "quantity" is a number with a dimension, like 1
    A "unit" is an amount of some dimension.
    A "dimension" is an attribute of a unit.

    """

    def __new__(cls, value: float, units: str | pd.Series, metadata: dict = {}):
        self = super().__new__(cls)

        self.metadata = metadata

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

            if np.isfinite(self.base_units_value).any():
                fid_x = lazy_nanquantile(np.abs(self.base_units_value), q=0.99)

                if fid_x > 0:
                    total_factor = 1
                    units = self.base_units
                    units_loss = np.inf

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

    def __repr__(self, prec: int = None) -> str:
        repr_spec = self.metadata.get("repr_spec")

        if self.physical_quantity == "time":
            if self.ndim == 0:
                if repr_spec == "date":
                    return self.date
                if np.any(self.s > 3600):
                    return self.ydhms
        if self.size > 1 or self.ndim > 0 or (prec is not None):
            prec = prec if prec is not None else compute_resolution_precision(self.human_value)
            value_repr = np.round(self.human_value, prec)
        else:
            value_repr = f"{self.human_value:.04g}"
        return f"{value_repr}{UNITS['symbol'].get(self.human_units) or f' {self.human_units}'}"

    def __neg__(self):
        return type(self)(-self.base_units_value, units=self.dimension_vector)

    def __add__(self, other):
        if (self.dimension_vector == other.dimension_vector).all():
            return type(self)(self.base_units_value + other.base_units_value, units=self.dimension_vector)
        else:
            raise UnitError(f"Cannot add units {self} and {other}")

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
        return type(self)(self.base_units_value**power, units=self.dimension_vector * power, metadata=self.metadata)

    # def __iter__(self):
    #     u = self.units
    #     for x in range(self.value):
    #         yield Quantity(x, units=u)

    def __copy__(self):
        return Quantity(self.base_units_value, self.dimension_vector, metadata=self.metadata)

    def __deepcopy__(self, memo):
        return self.__copy__()

    @property
    def ydhms(self) -> str:
        if self.physical_quantity != "time":
            raise ValueError("'ydhms' is only for times")
        parts = []
        t = self.seconds
        for k, v in {"y": 365 * 86400, "d": 86400, "h": 3600, "m": 60}.items():
            if t > v:
                parts.append(f"{int(t // v)}{k}")
                t = t % v
        parts.append(f"{t:.03f}s")
        return " ".join(parts)

    @property
    def dms(self):
        if self.physical_quantity != "angle":
            raise UnitError("Attribute 'dms' can only be computed for angles")
        sign, d, m, s = deg_to_signed_dms(self.deg)
        return f"{int(sign * d):>02}°{int(m):>02}’{s:.02f}”"

    @property
    def hms(self):
        if self.physical_quantity != "angle":
            raise UnitError("Attribute 'hms' can only be computed for angles")
        sign, h, m, s = deg_to_signed_hms(self.deg)
        return f"{int(sign * h):>02}ʰ{int(m):>02}ᵐ{s:.02f}ˢ"

    @property
    def date(self):
        if self.physical_quantity != "time":
            raise UnitError("Attribute 'date' can only be computed for angles")
        return arrow.get(self.seconds).isoformat(sep=" ", timespec="milliseconds")

    def repr_angle(self, format: str) -> str:
        if format == "dms":
            if self.physical_quantity != "angle":
                raise ValueError("string format 'dms' is only for angles")

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
        return Quantity(
            np.mean(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units, metadata=self.metadata
        )

    def median(self, axis=None, *args, **kwargs):
        return Quantity(
            np.median(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units, metadata=self.metadata
        )

    def min(self, axis=None, *args, **kwargs):
        return Quantity(
            np.min(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units, metadata=self.metadata
        )

    def max(self, axis=None, *args, **kwargs):
        return Quantity(
            np.max(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units, metadata=self.metadata
        )

    def std(self, axis=None, *args, **kwargs):
        return Quantity(
            np.std(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units, metadata=self.metadata
        )

    def ptp(self, axis=None, *args, **kwargs):
        return Quantity(
            np.ptp(self.base_units_value, axis=axis, *args, **kwargs), units=self.base_units, metadata=self.metadata
        )

    @property
    def shape(self):
        return np.shape(self.base_units_value)

    def reshape(self, new_shape):
        return Quantity(self.base_units_value.reshape(new_shape), self.base_units, metadata=self.metadata)

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
        return Quantity(self.base_units_value[key], units=self.base_units, metadata=self.metadata)

    def __format__(self, *args, **kwargs):
        return repr(self).__format__(*args, **kwargs)

    def __bool__(self):
        return self.size > 0

    def __len__(self):
        return len(self.base_units_value)

    def __neg__(self):
        return Quantity(-self.base_units_value, units=self.base_units, metadata=self.metadata)

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
