from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import scipy as sp
import dask.array as da

from ..tod import TOD
from .base import BaseMapper

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)


class BinMapper(BaseMapper):
    def __init__(
        self,
        center: tuple[float, float] = (0, 0),
        width: float = 1,
        height: float = None,
        resolution: float = 0.01,
        frame: str = "ra_dec",
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
            width=width,
            height=height,
            resolution=resolution,
            frame=frame,
            degrees=degrees,
            calibrate=calibrate,
            tods=tods,
            units=units,
        )

        self.tod_preprocessing = tod_preprocessing
        self.map_postprocessing = map_postprocessing

    def _run(self, band):
        """
        The actual mapper for the BinMapper.
        """

        band_map_data = {
            "sum": da.zeros((self.n_y, self.n_x)),
            "weight": da.zeros((self.n_y, self.n_x)),
        }

        # tods_pbar = tqdm(
        #     self.tods,
        #     desc=f"Running mapper ({band})",
        #     disable=not self.verbose,
        # )  # noqa

        for tod in self.tods:

            band_tod = (
                tod.subset(band=band.name)
                .process(config=self.tod_preprocessing)
                .to(self.units)
            )

            dx, dy = band_tod.coords.offsets(frame=self.frame, center=self.center)

            nu = band_tod.dets.band_center.mean()

            map_sum = sp.stats.binned_statistic_2d(
                dx.ravel(),
                dy.ravel(),
                (band_tod.weight * band_tod.signal).ravel(),
                bins=(self.x_bins, self.y_bins),
                statistic="sum",
            )[0]

            map_weight = sp.stats.binned_statistic_2d(
                dx.ravel(),
                dy.ravel(),
                band_tod.weight.ravel(),
                bins=(self.x_bins, self.y_bins),
                statistic="sum",
            )[0]

            del band_tod

            band_map_data["sum"] += map_sum
            band_map_data["weight"] += map_weight

        band_map_data["nom_freq"] = nu

        return band_map_data
