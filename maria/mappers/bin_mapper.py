from __future__ import annotations

import logging
import os
import time as ttime
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from ..io import humanize_time
from ..tod import TOD
from .base import BaseProjectionMapper

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class BinMapper(BaseProjectionMapper):
    def __init__(
        self,
        center: tuple[float, float] = None,
        stokes: str = "I",
        width: float = 1,
        height: float = None,
        resolution: float = 0.01,
        frame: str = "ra/dec",
        units: str = "K_RJ",
        degrees: bool = True,
        calibrate: bool = False,
        tod_preprocessing: dict = {},
        map_postprocessing: dict = {},
        tods: Sequence[TOD] = [],
    ):
        height = height or width

        super().__init__(
            center=center,
            stokes=stokes,
            width=width,
            height=height,
            resolution=resolution,
            frame=frame,
            degrees=degrees,
            calibrate=calibrate,
            tods=tods,
            units=units,
            tod_preprocessing=tod_preprocessing,
            map_postprocessing=map_postprocessing,
        )

    def initialize_mapper(self):
        return

    def _run(self):
        """
        The actual mapper for the BinMapper.
        """

        self.map_data = {}

        map_sum = np.zeros((self.n_stokes, len(self.bands), (self.n_y + 2) * (self.n_x + 2)))
        map_wgt = np.zeros((self.n_stokes, len(self.bands), (self.n_y + 2) * (self.n_x + 2)))
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

                P = self.map.pointing_matrix(coords=band_tod.coords)
                D = band_tod.signal.compute()
                W = band_tod.weight.compute()

                stokes_weight = band_tod.dets.stokes_weight()

                for stokes_index, stokes in enumerate(self.stokes):
                    pbar.set_postfix(tod=f"{tod_index + 1}/{len(self.tods)}", stokes=stokes)

                    s = stokes_weight[:, "IQUV".index(stokes)][..., None]

                    map_sum[stokes_index, band_index] += P @ (np.sign(s) * W * D).ravel()
                    map_wgt[stokes_index, band_index] += P @ (np.abs(s) * W).ravel()

                del band_tod

            logger.info(f"Ran mapper for band {band.name} in {humanize_time(ttime.monotonic() - band_start_s)}.")

        return {
            "sum": map_sum.reshape(self.n_stokes, self.n_bands, self.n_y + 2, self.n_x + 2)[..., 1:-1, 1:-1],
            "wgt": map_wgt.reshape(self.n_stokes, self.n_bands, self.n_y + 2, self.n_x + 2)[..., 1:-1, 1:-1],
            "stokes": self.stokes,
            "nu": nu_list,
        }
