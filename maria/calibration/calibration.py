import os

import pandas as pd

from ..atmosphere import AtmosphericSpectrum
from ..units import parse_units
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
    spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature,
)

here, this_filename = os.path.split(__file__)

transition_dict = {}
transition_dict["brightness_temperature"] = {"radiant_flux": brightness_temperature_to_radiant_flux}

transition_dict["radiant_flux"] = {
    "rayleigh_jeans_temperature": radiant_flux_to_rayleigh_jeans_temperature,
    "cmb_temperature_anisotropy": radiant_flux_to_cmb_temperature_anisotropy,
    "brightness_temperature": radiant_flux_to_brightness_temperature,
}
transition_dict["rayleigh_jeans_temperature"] = {
    "radiant_flux": rayleigh_jeans_temperature_to_radiant_flux,
    "cmb_temperature_anisotropy": rayleigh_jeans_temperature_to_cmb_temperature_anisotropy,
    "spectral_flux_density_per_pixel": rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel,
}

transition_dict["cmb_temperature_anisotropy"] = {
    "radiant_flux": cmb_temperature_anisotropy_to_radiant_flux,
    "rayleigh_jeans_temperature": cmb_temperature_anisotropy_to_rayleigh_jeans_temperature,
}

transition_dict["spectral_flux_density_per_pixel"] = {
    "rayleigh_jeans_temperature": spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature
}

quantities = list(transition_dict.keys())
for q in quantities:
    transition_dict[q][q] = identity


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
                if transition_dict.get(walk[-1], {}).get(end_node):
                    extended_walk = [*walk, end_node]
                    extended_walks.append(extended_walk)
                    if not paths_dict[start_node][end_node]:
                        function_path = [
                            transition_dict[_q2][_q1] for _q1, _q2 in zip(extended_walk[:-1], extended_walk[1:])
                        ]
                        paths_dict[start_node][end_node] = function_path[::-1]
        walks_dict[start_node] = extended_walks
    steps += 1

function_chains = pd.DataFrame(paths_dict)


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
                "pixel_area",
                "band",
                "spectrum",
                "zenith_pwv",
                "base_temperature",
                "elevation",
            ]:
                raise ValueError(f"Invalid kwarg '{key}'.")

    def __call__(self, x) -> float:
        if self.config.loc["quantity", "in"] == self.config.loc["quantity", "out"]:
            return x * self.in_factor / self.out_factor

        y = x * self.in_factor

        for f in self.function_chain():
            y = f(y, **self.kwargs)

        return y / self.out_factor

    def function_chain(self):
        try:
            return function_chains.loc[self.in_quantity, self.out_quantity]
        except KeyError:
            raise ValueError(f"Cannot convert from {self.in_quantity} to {self.out_quantity}")

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

    def __repr__(self):
        stuffing = ", ".join([self.signature, *[f"{k}={v}" for k, v in self.kwargs.items()]])
        return f"Calibration({stuffing})"
