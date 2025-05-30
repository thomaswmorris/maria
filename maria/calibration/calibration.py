import os

import pandas as pd
import yaml

from ..atmosphere import AtmosphericSpectrum
from ..io import leftpad
from ..units import QUANTITIES, Quantity, parse_units
from .conversion import (
    brightness_temperature_to_radiant_flux,
    cmb_temperature_anisotropy_to_radiant_flux,
    cmb_temperature_anisotropy_to_rayleigh_jeans_temperature,
    identity,
    radiant_flux_to_brightness_temperature,
    radiant_flux_to_cmb_temperature_anisotropy,
    radiant_flux_to_rayleigh_jeans_temperature,
    rayleigh_jeans_temperature_to_cmb_temperature_anisotropy,
    rayleigh_jeans_temperature_to_radiant_flux,
    rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel,
    spectral_flux_density_per_beam_to_spectral_flux_density_per_pixel,
    spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature,
    spectral_flux_density_per_pixel_to_spectral_flux_density_per_beam,
    spectral_flux_density_per_pixel_to_spectral_radiance,
    spectral_radiance_to_spectral_flux_density_per_pixel,
)

here, this_filename = os.path.split(__file__)

conversions = {}
conversions["brightness_temperature"] = {"radiant_flux": {"f": brightness_temperature_to_radiant_flux, "linear": False}}

conversions["radiant_flux"] = {
    "rayleigh_jeans_temperature": {"f": radiant_flux_to_rayleigh_jeans_temperature, "linear": True},
    "cmb_temperature_anisotropy": {"f": radiant_flux_to_cmb_temperature_anisotropy, "linear": True},
    "brightness_temperature": {"f": radiant_flux_to_brightness_temperature, "linear": False},
}
conversions["rayleigh_jeans_temperature"] = {
    "radiant_flux": {"f": rayleigh_jeans_temperature_to_radiant_flux, "linear": True},
    "cmb_temperature_anisotropy": {"f": rayleigh_jeans_temperature_to_cmb_temperature_anisotropy, "linear": False},
    "spectral_flux_density_per_pixel": {"f": rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel, "linear": False},
}

conversions["cmb_temperature_anisotropy"] = {
    "radiant_flux": {"f": cmb_temperature_anisotropy_to_radiant_flux, "linear": True},
    "rayleigh_jeans_temperature": {"f": cmb_temperature_anisotropy_to_rayleigh_jeans_temperature, "linear": False},
}

conversions["spectral_flux_density_per_pixel"] = {
    "rayleigh_jeans_temperature": {"f": spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature, "linear": False},
    "spectral_radiance": {"f": spectral_flux_density_per_pixel_to_spectral_radiance, "linear": True},
    "spectral_flux_density_per_beam": {
        "f": spectral_flux_density_per_pixel_to_spectral_flux_density_per_beam,
        "linear": True,
    },
}

conversions["spectral_flux_density_per_beam"] = {
    "spectral_flux_density_per_pixel": {
        "f": spectral_flux_density_per_beam_to_spectral_flux_density_per_pixel,
        "linear": True,
    },
}

conversions["spectral_radiance"] = {
    "spectral_flux_density_per_pixel": {"f": spectral_radiance_to_spectral_flux_density_per_pixel, "linear": True},
}

quantities = list(conversions.keys())
for q in quantities:
    conversions[q][q] = identity


walks_dict = {}
paths_dict = {}
for q1 in quantities:
    walks_dict[q1] = [[q1]]
    paths_dict[q1] = {q: [] for q in quantities}

max_steps = 10

steps = 0
while any([paths_dict[q1][q2] == [] for q2 in quantities for q1 in quantities]) and steps < max_steps:
    for start_node in quantities:
        extended_walks = []
        for walk in walks_dict[start_node]:
            for end_node in quantities:
                if conversions.get(walk[-1], {}).get(end_node):
                    extended_walk = [*walk, end_node]
                    extended_walks.append(extended_walk)
                    if not paths_dict[start_node][end_node]:
                        function_path = [conversions[_q2][_q1] for _q1, _q2 in zip(extended_walk[:-1], extended_walk[1:])]
                        # paths_dict[start_node][end_node] = function_path[::-1]
                        paths_dict[start_node][end_node] = extended_walk[::-1]
        walks_dict[start_node] = extended_walks
    steps += 1

calibration_chains = pd.DataFrame(paths_dict)


def parse_calibration_signature(s: str):
    res = {}
    for sep in ["->"]:
        if s.count(sep) == 1:
            if sep is not None:
                items = [u.strip() for u in s.split(sep)]
                if len(items) == 2:
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

    def qchain(self):
        try:
            return calibration_chains.loc[self.in_quantity, self.out_quantity]
        except KeyError:
            raise ValueError(f"Cannot convert from {self.in_quantity} to {self.out_quantity}")

    def uchain(self):
        middle_terms = [QUANTITIES[q]["default_unit"] for q in self.qchain()[1:-1]]
        return " -> ".join([self.in_units, *middle_terms, self.out_units])

    def function_chain(self):
        qchain = self.qchain()
        return [conversions[q1][q2]["f"] for q1, q2 in zip(qchain[:-1], qchain[1:])]

    def linear(self):
        qchain = self.qchain()
        return all([conversions[q2][q1]["linear"] for q1, q2 in zip(qchain[:-1], qchain[1:])])

    def __call__(self, x) -> float:
        if self.config.loc["quantity", "in"] == self.config.loc["quantity", "out"]:
            return x * self.in_factor / self.out_factor

        y = x * self.in_factor

        for f in self.function_chain():
            y = f(y, **self.kwargs)

        return y / self.out_factor

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
        return self.config.loc["quantity", "in"]

    @property
    def out_quantity(self) -> float:
        return self.config.loc["quantity", "out"]

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
            if k in ["spectrum", "band"]:
                continue
            qkwargs[k] = str(Quantity(v, KWARGS_UNITS[k]))

        # spectrum_string = self.kwargs.get("spectrum") or ""
        # spectrum_string = self.kwargs.get("spectrum") or ""
        # kwargs_string = yaml.dump(qkwargs)

        factor = f"{self(1e0):.03e}" if self.linear() else "None (nonlinear)"

        return f"""Calibration({self.signature}):
  factor: {factor}
  chain: {self.uchain()}
  spectrum: {self.kwargs.get("spectrum")}
  band: {self.kwargs.get("band")}
  kwargs: {qkwargs}"""

    # def __repr__(self):
    #     filling = copy.deepcopy(self.kwargs.items())
    #     fill_string = ", ".join([self.signature, *[f"{k}={v}" for k, v in self.kwargs.items()]])
    #     return f"Calibration({stuffing})"
