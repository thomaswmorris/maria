from __future__ import annotations

import logging
import os
import time as ttime
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from ..coords import infer_center_width_height
from ..io import humanize_time, repr_phi_theta
from ..map import ProjectionMap
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
    ):
        super().__init__(
            tods=tods,
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
        )

    def _run(self):
        """
        The actual mapper for the BinMapper.
        """

        map_sum = np.zeros(self.map_size)
        map_wgt = np.zeros(self.map_size)

        pbar = tqdm(
            self.tods,
            total=len(self.tods),
            desc=f"Mapping",
            postfix={"tod": f"1/{len(self.tods)}"},
        )

        for tod in pbar:
            if not tod.shape[0] > 0:
                continue

            P = self.map.pointing_matrix(coords=tod.coords, dets=tod.dets)
            D = tod.signal.compute().ravel()
            W = tod.weight.compute().ravel()

            # stokes_weight = band_tod.dets.stokes_weight()

            # for stokes_index, stokes in enumerate(self.stokes):
            #     pbar.set_postfix(tod=f"{tod_index + 1}/{len(self.tods)}", stokes=stokes)

            #     s = stokes_weight[:, "IQUV".index(stokes)][..., None]

            map_sum += (W * D) @ P
            map_wgt += W @ np.abs(P)

            # del band_tod

            # logger.info(f"Ran mapper for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

        return {
            "data": (map_sum / map_wgt).reshape(self.map_shape),
            "weight": map_wgt.reshape(self.map_shape),
            "sum": map_sum.reshape(self.map_shape),
        }
