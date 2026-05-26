import os

import pandas as pd
import yaml

from ..errors import MissingConversionParametersError
from ..spectrum import AtmosphericSpectrum
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


KWARGS_UNITS = {
    "nu": "Hz",
    "pixel_area": "sr",
    "beam_area": "sr",
    "zenith_pwv": "mm",
    "base_temperature": "K",
    "elevation": "rad",
}


def compute_quantities_chain(
    start_quantity, end_quantity, max_steps: int = 6, kwargs: dict = {}, enforce_kwargs: bool = True
):
    """
    What the fuck
    """
    missing_kwargs = None
    walks_and_kwargs = [([start_quantity], set())]
    for _ in range(max_steps):
        extended_walks_and_kwargs = []
        while len(walks_and_kwargs):
            walk, walk_required_kwargs = walks_and_kwargs.pop(0)
            for quantity, quantity_config in conversions.get(walk[-1], {}).items():
                required_kwargs = set(quantity_config.get("required_kwargs", [])) | walk_required_kwargs
                chain = [*walk, quantity]

                if quantity == end_quantity:
                    if all([kwarg in kwargs for kwarg in required_kwargs]) or not enforce_kwargs:
                        return chain
                    if missing_kwargs is None:
                        missing_kwargs = required_kwargs

                if quantity not in walk:
                    extended_walks_and_kwargs.append((chain, required_kwargs))

        walks_and_kwargs = extended_walks_and_kwargs

    raise MissingConversionParametersError(
        f"Conversion from '{start_quantity}' to '{end_quantity}' is missing kwargs {missing_kwargs}"
    )


class Calibration:
    def __init__(self, signature: str, spectrum: AtmosphericSpectrum = None, **kwargs):
        if not isinstance(signature, str):
            raise ValueError("'signature' must be a string.")

        self.config = pd.DataFrame(parse_calibration_signature(signature))
        self.signature = signature
        self.kwargs = {"spectrum": spectrum, **kwargs}

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

    def linear(self):
        qchain = compute_quantities_chain(self.in_quantity, self.out_quantity, enforce_kwargs=False)
        return all([conversions[q1][q2]["linear"] for q1, q2 in zip(qchain[:-1], qchain[1:])])

    def __call__(self, x, **kwargs) -> float:
        y = Quantity(x, self.in_units).base_units_value

        calibration_kwargs = self.kwargs.copy()
        calibration_kwargs.update(kwargs)
        quantities_chain = compute_quantities_chain(self.in_quantity, self.out_quantity, kwargs=calibration_kwargs)

        for q1, q2 in zip(quantities_chain[:-1], quantities_chain[1:]):
            y = conversions[q1][q2]["f"](y, **calibration_kwargs)

        return Quantity(y, QUANTITY_DIMENSION_VECTORS.loc[quantities_chain[-1]]).to(self.out_units)

    @property
    def in_units(self) -> float:
        return self.config.loc["units", "in"]

    @property
    def out_units(self) -> float:
        return self.config.loc["units", "out"]

    @property
    def in_factor(self) -> float:
        return self.config.loc["factor", "in"]

    @property
    def out_factor(self) -> float:
        return self.config.loc["factor", "out"]

    @property
    def in_quantity(self) -> float:
        return self.config.loc["physical_quantity", "in"]

    @property
    def out_quantity(self) -> float:
        return self.config.loc["physical_quantity", "out"]

    @property
    def in_to_K_RJ(self) -> float:
        return self.config.loc["from", "in"]

    @property
    def K_RJ_to_out(self) -> float:
        return self.config.loc["to", "out"]

    def leftpad(thing, n: int = 2, char=" "):
        return "\n".join([n * char + line for line in str(thing).splitlines()])

    def __repr__(self):
        KWARGS_UNITS = {"nu": "Hz", "pixel_area": "sr", "zenith_pwv": "mm", "base_temperature": "K", "elevation": "rad"}

        qkwargs = {}
        for k, v in self.kwargs.items():
            if k in ["spectrum", "band", "polarized"]:
                continue
            qkwargs[k] = str(Quantity(v, KWARGS_UNITS[k]))

        if not hasattr(self, "_factor"):
            self._factor = f"{self(1e0):.03e}" if self.linear() else "None (nonlinear)"

        return f"""Calibration({self.signature}):
  spectrum: {self.kwargs.get("spectrum")}
  band: {self.kwargs.get("band")}
  kwargs: {qkwargs}"""

    # def __repr__(self):
    #     filling = copy.deepcopy(self.kwargs.items())
    #     fill_string = ", ".join([self.signature, *[f"{k}={v}" for k, v in self.kwargs.items()]])
    #     return f"Calibration({stuffing})"
