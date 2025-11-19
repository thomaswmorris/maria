from __future__ import annotations

import logging
import os
import time as ttime
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from ..coords import infer_center_width_height
from ..io import humanize_time, repr_phi_theta
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
        calibrate: bool = False,
        min_time: float = None,
        max_time: float = None,
        timestep: float = None,
        tod_preprocessing: dict = {},
        map_postprocessing: dict = {},
    ):
        if not tods:
            raise ValueError("You must pass at least one TOD to the mapper!")

        super().__init__(
            stokes=stokes,
            center=center,
            width=width,
            height=height,
            resolution=resolution,
            frame=frame,
            calibrate=calibrate,
            tods=tods,
            units=units,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
            min_time=min_time,
            max_time=max_time,
            timestep=timestep,
            degrees=degrees,
        )

    def initialize_mapper(self):
        return

    def _run(self):
        """
        The actual mapper for the BinMapper.
        """

        self.map_data = {}

        map_sum = np.zeros((self.n_stokes, len(self.map.nu), len(self.map.t) * self.n_y * self.n_x))
        map_wgt = np.zeros((self.n_stokes, len(self.map.nu), len(self.map.t) * self.n_y * self.n_x))

        nu_list = []

        for band_index, band in enumerate(self.bands):
            band_start_s = ttime.monotonic()

            nu_list.append(band.center)

            pbar = tqdm(
                enumerate(self.tods),
                total=len(self.tods),
                desc=f"Mapping band {band.name}",
                postfix={"tod": f"1/{len(self.tods)}", "stokes": "I"},
            )

            for tod_index, tod in pbar:
                band_tod = tod.subset(band=band.name)

                if not tod.shape[0] > 0:
                    continue

                pmat = self.map.pointing_matrix(coords=band_tod.coords)
                D = band_tod.signal.compute()
                W = band_tod.weight.compute()

                stokes_weight = band_tod.dets.stokes_weight()

                for stokes_index, stokes in enumerate(self.stokes):
                    pbar.set_postfix(tod=f"{tod_index + 1}/{len(self.tods)}", stokes=stokes)

                    s = stokes_weight[:, "IQUV".index(stokes)][..., None]

                    map_sum[stokes_index, band_index] += (np.sign(s) * W * D).ravel() @ pmat
                    map_wgt[stokes_index, band_index] += (np.abs(s) * W).ravel() @ pmat

                del band_tod

            logger.info(f"Ran mapper for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

        return {
            "sum": map_sum.reshape(self.n_stokes, self.n_bands, len(self.map.t), self.n_y, self.n_x),
            "wgt": map_wgt.reshape(self.n_stokes, self.n_bands, len(self.map.t), self.n_y, self.n_x),
            "stokes": self.stokes,
            "nu": nu_list,
        }
