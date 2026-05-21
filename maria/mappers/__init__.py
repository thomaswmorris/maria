from __future__ import annotations

from ..map import Map
from .bin_mapper import BinMapper  # noqa
from .ml_mapper import MaximumLikelihoodMapper  # noqa


def compute_residual_map(input_map: Map, output_map: Map):

    in_map = input_map.copy()

    for dim in output_map.dims:
        if dim not in in_map.dims:
            in_map.unsqueeze(dim)

    residual_map = output_map.resample(in_map).to(in_map.units)
    residual_map.data -= in_map.data

    return residual_map
