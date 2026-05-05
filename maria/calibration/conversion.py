import os

import pandas as pd

from .functions import *  # noqa

here, this_filename = os.path.split(__file__)

conversions = {}
conversions["brightness_temperature"] = {
    "power": {"f": brightness_temperature_to_power, "linear": False, "required_kwargs": ["band"]},
    "cmb_temperature_anisotropy": {"f": brightness_temperature_to_cmb_temperature_anisotropy, "linear": False},
}

conversions["power"] = {
    "rayleigh_jeans_temperature": {"f": power_to_rayleigh_jeans_temperature, "linear": True, "required_kwargs": ["band"]},
    "cmb_temperature_anisotropy": {"f": power_to_cmb_temperature_anisotropy, "linear": True, "required_kwargs": ["band"]},
    "brightness_temperature": {"f": power_to_brightness_temperature, "required_kwargs": ["band"]},
}
conversions["rayleigh_jeans_temperature"] = {
    "power": {"f": rayleigh_jeans_temperature_to_power, "linear": True, "required_kwargs": ["band"]},
    "cmb_temperature_anisotropy": {
        "f": rayleigh_jeans_temperature_to_cmb_temperature_anisotropy,
        "linear": False,
        "required_kwargs": ["nu"],
    },
    "spectral_flux_density_per_pixel": {
        "f": rayleigh_jeans_temperature_to_spectral_flux_density_per_pixel,
        "required_kwargs": ["nu"],
    },
}

conversions["cmb_temperature_anisotropy"] = {
    "power": {"f": cmb_temperature_anisotropy_to_power, "linear": True, "required_kwargs": ["band"]},
    "brightness_temperature": {"f": cmb_temperature_anisotropy_to_brightness_temperature, "linear": False},
    "rayleigh_jeans_temperature": {
        "f": cmb_temperature_anisotropy_to_rayleigh_jeans_temperature,
        "linear": False,
        "required_kwargs": ["nu"],
    },
    "compton_y": {"f": cmb_temperature_anisotropy_to_compton_y, "linear": False, "required_kwargs": ["nu"]},
}

conversions["spectral_flux_density_per_pixel"] = {
    "rayleigh_jeans_temperature": {
        "f": spectral_flux_density_per_pixel_to_rayleigh_jeans_temperature,
        "linear": False,
        "required_kwargs": ["nu"],
    },
    "spectral_radiance": {
        "f": spectral_flux_density_per_pixel_to_spectral_radiance,
        "linear": True,
        "required_kwargs": ["nu"],
    },
    "spectral_flux_density_per_beam": {
        "f": spectral_flux_density_per_pixel_to_spectral_flux_density_per_beam,
        "linear": True,
        "required_kwargs": ["nu"],
    },
}

conversions["spectral_flux_density_per_beam"] = {
    "spectral_flux_density_per_pixel": {
        "f": spectral_flux_density_per_beam_to_spectral_flux_density_per_pixel,
        "linear": True,
        "required_kwargs": ["nu"],
    },
}

conversions["spectral_radiance"] = {
    "spectral_flux_density_per_pixel": {
        "f": spectral_radiance_to_spectral_flux_density_per_pixel,
        "linear": True,
        "required_kwargs": ["nu"],
    },
}

conversions["compton_y"] = {
    "cmb_temperature_anisotropy": {"f": compton_y_to_cmb_temperature_anisotropy, "linear": False, "required_kwargs": ["nu"]},
}

# quantities = list(conversions.keys())
# for q in quantities:
#     conversions[q][q] = {"f": identity, "linear": True}

# walks_dict = {}
# paths_dict = {}
# for q1 in quantities:
#     walks_dict[q1] = [[q1]]
#     paths_dict[q1] = {q: [] for q in quantities}

# max_steps = 10

# steps = 0
# while any([paths_dict[q1][q2] == [] for q2 in quantities for q1 in quantities]) and steps < max_steps:
#     for start_node in quantities:
#         extended_walks = []
#         for walk in walks_dict[start_node]:
#             for end_node in quantities:
#                 if conversions.get(walk[-1], {}).get(end_node):
#                     extended_walk = [*walk, end_node]
#                     extended_walks.append(extended_walk)
#                     if not paths_dict[start_node][end_node]:
#                         function_path = [conversions[_q2][_q1] for _q1, _q2 in zip(extended_walk[:-1], extended_walk[1:])]
#                         # paths_dict[start_node][end_node] = function_path[::-1]
#                         paths_dict[start_node][end_node] = extended_walk[::-1]
#         walks_dict[start_node] = extended_walks
#     steps += 1

# conversion_chains = pd.DataFrame(paths_dict)
