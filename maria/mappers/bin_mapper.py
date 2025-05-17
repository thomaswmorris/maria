from __future__ import annotations

import os
from collections.abc import Sequence

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..map import ProjectedMap
from ..tod import TOD
from .base import BaseMapper

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)


class BinMapper(BaseMapper):
    def __init__(
        self,
        center: tuple[float, float] = (0, 0),
        stokes: str = "I",
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
            stokes=stokes,
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

        self.n_x = int(np.maximum(1, self.width / self.resolution))
        self.n_y = int(np.maximum(1, self.height / self.resolution))

        self.x_bins = np.linspace(-0.5 * self.width, 0.5 * self.width, self.n_x + 1)
        self.y_bins = np.linspace(0.5 * self.height, -0.5 * self.height, self.n_y + 1)

    def _run(self, band):
        """
        The actual mapper for the BinMapper.
        """

        # nu = np.unique([band.center for tod in self.tods for band in tod.dets.bands])

        self.map = ProjectedMap(
            data=np.zeros((self.n_y, self.n_x)),
            width=self.width,
            center=self.center,
            frame=self.frame,
            degrees=False,
        )

        map_numer = np.zeros((self.n_stokes, (self.n_y + 2) * (self.n_x + 2)))
        map_denom = np.zeros((self.n_stokes, (self.n_y + 2) * (self.n_x + 2)))

        for tod in self.tods:
            band_tod = tod.subset(band=band.name)

            if not tod.shape[0] > 0:
                continue

            band_tod = band_tod.process(config=self.tod_preprocessing).to(self.units)

            P = self.map.pointing_matrix(coords=band_tod.coords)
            D = band_tod.signal.compute()
            W = band_tod.weight.compute()

            stokes_weight = band_tod.dets.stokes_weight()

            stokes_pbar = tqdm(enumerate(self.stokes), total=self.n_stokes, desc=f"Mapping band {band.name}")

            for stokes_index, stokes in stokes_pbar:
                stokes_pbar.set_postfix(band=band.name, stokes=stokes)

                s = stokes_weight[:, "IQUV".index(stokes)][..., None]

                map_numer[stokes_index] += P @ (np.sign(s) * W * D).ravel()
                map_denom[stokes_index] += P @ (np.abs(s) * W).ravel()

                # band_map_data["sum"][stokes_index] += sp.stats.binned_statistic_2d(
                #     dy.ravel(),
                #     dx.ravel(),
                #     (np.sign(w) * band_tod.weight * band_tod.signal).ravel(),
                #     bins=(self.y_bins[::-1], self.x_bins),
                #     statistic="sum",
                # ).statistic[::-1]

                # band_map_data["weight"][stokes_index] += sp.stats.binned_statistic_2d(
                #     dy.ravel(),
                #     dx.ravel(),
                #     (np.abs(w) * band_tod.weight).ravel(),
                #     bins=(self.y_bins[::-1], self.x_bins),
                #     statistic="sum",
                # ).statistic[::-1]

            del band_tod

        band_map_data = {
            "numer": map_numer.reshape(self.n_stokes, self.n_y + 2, self.n_x + 2)[:, 1:-1, 1:-1],
            "denom": map_denom.reshape(self.n_stokes, self.n_y + 2, self.n_x + 2)[:, 1:-1, 1:-1],
            "nom_freq": band.center,
        }

        return band_map_data
