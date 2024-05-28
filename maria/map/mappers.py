import os
from typing import Sequence, Tuple

import numpy as np
import scipy as sp
from todder import TOD
from tqdm import tqdm

from .. import utils
from .map import Map

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)


class BaseMapper:
    """
    The base class for mapping.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0, 0),
        width: float = 1,
        height: float = 1,
        resolution: float = 0.01,
        frame: str = "ra_dec",
        units: str = "K_RJ",
        degrees: bool = True,
        calibrate: bool = False,
        tods: Sequence[TOD] = [],
        verbose: bool = True,
    ):
        self.resolution = np.radians(resolution) if degrees else resolution
        self.center = np.radians(center) if degrees else center
        self.width = np.radians(width) if degrees else width
        self.height = np.radians(height) if degrees else height
        self.degrees = degrees
        self.calibrate = calibrate
        self.frame = frame
        self.units = units

        self.verbose = verbose

        self.n_x = int(np.maximum(1, self.width / self.resolution))
        self.n_y = int(np.maximum(1, self.height / self.resolution))

        self.x_bins = np.linspace(-0.5 * self.width, 0.5 * self.width, self.n_x + 1)
        self.y_bins = np.linspace(-0.5 * self.height, 0.5 * self.height, self.n_y + 1)

        self.tods = []
        self.add_tods(tods)

    def plot(self):
        if not hasattr(self, "map"):
            raise RuntimeError("Mapper has not been run yet.")
        self.map.plot()

    @property
    def n_maps(self):
        return len(self.maps)

    def add_tods(self, tods):
        for tod in np.atleast_1d(tods):
            self.tods.append(tod)
        self.bands = list(
            np.unique([list(np.unique(tod.dets.band_name)) for tod in self.tods])
        )

    def _run(self):
        raise ValueError("Not implemented!")

    def run(self):
        self.maps = {}

        for band in self.bands:
            self.maps[band] = self._run(band)

        return Map(
            data=np.concatenate([m.data for m in self.maps.values()], axis=0),
            weight=np.concatenate([m.weight for m in self.maps.values()], axis=0),
            resolution=self.resolution,
            frequency=[float(m.frequency) for m in self.maps.values()],
            center=self.center,
            degrees=False,
        )


class BinMapper(BaseMapper):
    def __init__(
        self,
        center: Tuple[float, float] = (0, 0),
        width: float = 1,
        height: float = 1,
        resolution: float = 0.01,
        frame: str = "ra_dec",
        units: str = "K_RJ",
        degrees: bool = True,
        calibrate: bool = False,
        tod_postprocessing: dict = {},
        map_postprocessing: dict = {},
        tods: Sequence[TOD] = [],
    ):
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

        self.tod_postprocessing = tod_postprocessing
        self.map_postprocessing = map_postprocessing

    def _run(self, band):
        """
        The actual mapper for the BinMapper.
        """

        self.band_data = {}

        self.raw_map_sums = np.zeros((self.n_x, self.n_y))
        self.raw_map_cnts = np.zeros((self.n_x, self.n_y))

        self.map_data = np.zeros((self.n_y, self.n_x))
        self.map_weight = np.zeros((self.n_y, self.n_x))

        tods_pbar = tqdm(
            self.tods, desc=f"Running mapper ({band})", disable=not self.verbose
        )

        for tod in tods_pbar:
            dx, dy = tod.coords.offsets(frame=self.frame, center=self.center)

            band_mask = tod.dets.band_name == band

            band_center = tod.dets.loc[band_mask, "band_center"].mean()

            D = tod.cal[band_mask, None] * tod.data[band_mask]

            # windowing
            W = np.ones(D.shape[0])[:, None] * sp.signal.windows.tukey(
                D.shape[-1], alpha=0.1
            )

            WD = W * sp.signal.detrend(D, axis=-1)
            del D

            if "highpass" in self.tod_postprocessing.keys():
                WD = utils.signal.highpass(
                    WD,
                    fc=self.tod_postprocessing["highpass"]["f"],
                    fs=tod.fs,
                    order=self.tod_postprocessing["highpass"].get("order", 1),
                    method="bessel",
                )

            if "remove_modes" in self.tod_postprocessing.keys():
                n_modes_to_remove = self.tod_postprocessing["remove_modes"]["n"]

                U, V = utils.signal.decompose(
                    WD, downsample_rate=np.maximum(int(tod.fs / 16), 1), mode="uv"
                )
                WD = U[:, n_modes_to_remove:] @ V[n_modes_to_remove:]

            if "despline" in self.tod_postprocessing.keys():
                B = utils.signal.get_bspline_basis(
                    tod.time.compute(),
                    spacing=self.tod_postprocessing["despline"].get("knot_spacing", 10),
                    order=self.tod_postprocessing["despline"].get("spline_order", 3),
                )

                A = np.linalg.inv(B @ B.T) @ B @ WD.T
                WD -= A.T @ B

            map_sum = sp.stats.binned_statistic_2d(
                dx[band_mask].ravel(),
                dy[band_mask].ravel(),
                WD.ravel(),
                bins=(self.x_bins, self.y_bins),
                statistic="sum",
            )[0]

            map_cnt = sp.stats.binned_statistic_2d(
                dx[band_mask].ravel(),
                dy[band_mask].ravel(),
                W.ravel(),
                bins=(self.x_bins, self.y_bins),
                statistic="sum",
            )[0]

            self.DATA = WD

            self.raw_map_sums += map_sum
            self.raw_map_cnts += map_cnt

            # self.band_data["band_center"] = tod.dets.band_center.mean()
            # self.band_data["band_width"] = 30

            band_map_numer = self.raw_map_sums.copy()
            band_map_denom = self.raw_map_cnts.copy()

            if "gaussian_filter" in self.map_postprocessing.keys():
                sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

                band_map_numer = sp.ndimage.gaussian_filter(band_map_numer, sigma=sigma)
                band_map_denom = sp.ndimage.gaussian_filter(band_map_denom, sigma=sigma)

            band_map_data = band_map_numer / band_map_denom

            mask = band_map_denom > 0

            if "median_filter" in self.map_postprocessing.keys():
                band_map_data = sp.ndimage.median_filter(
                    band_map_data, size=self.map_postprocessing["median_filter"]["size"]
                )

            self.map_data = np.where(mask, band_map_numer, np.nan) / np.where(
                mask, band_map_denom, 1
            )

            self.map_weight = band_map_denom

        return Map(
            data=self.map_data,
            weight=self.map_weight,
            resolution=self.resolution,
            frequency=band_center,
            center=self.center,
            degrees=False,
        )
