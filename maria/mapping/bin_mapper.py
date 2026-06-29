from __future__ import annotations

import logging
import os
import time as ttime
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from ..coords import infer_center_width_height
from ..io import DEFAULT_BAR_FORMAT, humanize_time, repr_phi_theta
from ..map import Map, ProjectionMap
from ..tod import TOD
from ..units import Quantity
from .base import BaseProjectionMapper

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BinMapper(BaseProjectionMapper):
    def __init__(
        self,
        tods: Sequence[TOD],
        target: Map = None,
        center: tuple[float, float] = None,
        stokes: str = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {},
        map_postprocessing: dict = {},
        progress_bars: bool = True,
        bilinear: bool = False,
    ):
        super().__init__(
            tods=tods,
            target=target,
            stokes=stokes,
            center=center,
            width=width,
            height=height,
            resolution=resolution,
            frame=frame,
            units=units,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            degrees=degrees,
            progress_bars=progress_bars,
            bilinear=bilinear,
        )

        self.products = {
            "data": np.zeros(self.map_shape),
            "weight": np.ones(self.map_shape),
        }

        self.has_been_ran = False

    @property
    def map(self):
        if not self.has_been_ran:
            raise RuntimeError("Mapper has not been run yet!")
        return super().map

    def get_map_data(self):
        return self.products["data"]

    def get_map_weight(self):
        return self.products["weight"]

    def run(self):
        """
        Run the BinMapper
        """

        map_sum = np.zeros(self.map_size)
        map_wgt = np.zeros(self.map_size)

        pbar = tqdm(
            self.tods,
            total=len(self.tods),
            desc=f"Mapping",
            postfix={"tod": f"1/{len(self.tods)}"},
            bar_format=DEFAULT_BAR_FORMAT,
            disable=not self.progress_bars,
        )

        for tod in pbar:
            if not tod.shape[0] > 0:
                continue

            P = super().map.stokes_weighted_pointing_matrix(coords=tod.coords, dets=tod.dets, bilinear=self.bilinear)
            D = tod.signal.compute().ravel()
            W = tod.weight.compute().ravel()

            map_sum += (W * D) @ P
            map_wgt += W @ np.abs(P)

        self.products = {
            "data": map_sum / map_wgt,
            "weight": map_wgt,
            "sum": map_sum,
        }

        self.has_been_ran = True

        return self.map
