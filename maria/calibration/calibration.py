import os

from ..errors import IncompatibleQuantityError
from ..units import QUANTITY_DIMENSION_VECTORS, Quantity, parse_units
from .conversion import conversions

here, this_filename = os.path.split(__file__)


def parse_calibration_signature(s: str):
    for sep in ["->"]:
        if s.count(sep) == 1:
            if sep is not None:
                items = [u.strip() for u in s.split(sep)]
                if len(items) == 2:
                    res = {}
                    for io, u in zip(["in", "out"], items):
                        res[io] = parse_units(u)
        return res
    raise ValueError("Calibration must have signature 'units1 -> units2'.")


def compute_quantities_chain(
    start_quantity,
    end_quantity,
    max_steps: int = 6,
    enforce_kwargs: bool = True,
    kwargs: dict = {},
):
    """
    Return a list of the chain of functions required to convert 'start_quantity' to 'end_quantity',
    and find the required kwargs (frequency, etc.)
    """

    walks_and_required_kwargs = [([start_quantity], set())]
    for _ in range(max_steps):
        extended_walks_and_required_kwargs = []
        while len(walks_and_required_kwargs):
            walk, required_kwargs = walks_and_required_kwargs.pop(0)
            for quantity, quantity_config in conversions.get(walk[-1], {}).items():
                quantity_required_kwargs = set(quantity_config.get("required_kwargs", []))

                if enforce_kwargs:
                    if not all([kwarg in kwargs for kwarg in quantity_required_kwargs]):
                        continue

                extended_walk = [*walk, quantity]
                extended_required_kwargs = quantity_required_kwargs | required_kwargs

                if quantity == end_quantity:
                    return extended_walk, list(extended_required_kwargs)

                if quantity not in walk:
                    extended_walks_and_required_kwargs.append((extended_walk, extended_required_kwargs))

        walks_and_required_kwargs = extended_walks_and_required_kwargs

    # if missing_kwargs is not None:
    #     raise MissingCalibrationKwargs(
    #         f"Conversion from '{start_quantity}' to '{end_quantity}' is missing kwargs {missing_kwargs}"
    #     )

    raise IncompatibleQuantityError(f"Cannot convert from quantity '{start_quantity}' to quantity '{end_quantity}'")


class Calibration:
    def __init__(self, signature: str, enforce_kwargs: bool = False, **kwargs):
        if not isinstance(signature, str):
            raise ValueError("'signature' must be a string.")

        self.config = parse_calibration_signature(signature)
        self.signature = signature
        self.kwargs = kwargs

        for key in kwargs:
            if key not in [
                "nu",
                "polarized",
                "pixel_area",
                "beam_area",
                "band",
                "spectrum",
                "zenith_pwv",
                "base_temperature",
                "elevation",
            ]:
                raise ValueError(f"Invalid kwarg '{key}'.")

        self.qchain, self.required_kwargs = compute_quantities_chain(
            self.in_quantity, self.out_quantity, kwargs=self.kwargs, enforce_kwargs=enforce_kwargs
        )

        self.is_linear = all([conversions[q1][q2]["linear"] for q1, q2 in zip(self.qchain[:-1], self.qchain[1:])])

        if self.is_linear:
            if all([kwarg in self.kwargs for kwarg in self.required_kwargs]):
                self.factor = self(1e0)

    def uchain(self):
        return " -> ".join([self.in_units, *[QUANTITIES.loc[q, "base_unit"] for q in self.qchain][1:-1], self.out_units])

    def __call__(self, x, **kwargs) -> float:

        if self.is_linear and hasattr(self, "factor"):
            return self.factor * x

        y = Quantity(x, self.in_units).base_units_value

        calibration_kwargs = self.kwargs.copy()
        calibration_kwargs.update(kwargs)

        for q1, q2 in zip(self.qchain[:-1], self.qchain[1:]):
            y = conversions[q1][q2]["f"](y, **calibration_kwargs)

        return Quantity(y, QUANTITY_DIMENSION_VECTORS.loc[self.qchain[-1]]).to(self.out_units)

    @property
    def in_units(self) -> str:
        return self.config["in"]["units"]

    @property
    def out_units(self) -> str:
        return self.config["out"]["units"]

    @property
    def in_factor(self) -> float:
        return self.config["in"]["factor"]

    @property
    def out_factor(self) -> float:
        return self.config["out"]["factor"]

    @property
    def in_quantity(self) -> str:
        return self.config["in"]["physical_quantity"]

    @property
    def out_quantity(self) -> str:
        return self.config["out"]["physical_quantity"]

    def leftpad(thing, n: int = 2, char=" "):
        return "\n".join([n * char + line for line in str(thing).splitlines()])

    def __repr__(self):

        if self.is_linear:
            if hasattr(self, "factor"):
                factor = self.factor
            else:
                factor = "missing kwargs"
        else:
            factor = "nonlinear"

        conversions = "\n    ".join([f"{q1} -> {q2}" for q1, q2 in zip(self.qchain[:-1], self.qchain[1:])])

        return f"""Calibration({self.signature}):
  factor: {factor}
  in:
    units: {self.in_units}
    quantity: {self.in_quantity}
  out:
    units: {self.out_units}
    quantity: {self.out_quantity}
  conversions:
    {conversions}
  kwargs:
    required: {self.required_kwargs}
    supplied: {self.kwargs}"""
