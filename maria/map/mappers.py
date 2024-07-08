import os
from typing import Sequence, Tuple

import numpy as np
import scipy as sp
from tqdm import tqdm

from ..tod import TOD
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
        self.map_data = {}

        for band in self.bands:
            self.map_data[band] = self._run(band)

        map_data = np.zeros((len(self.map_data), self.n_y, self.n_x))
        map_weight = np.zeros((len(self.map_data), self.n_y, self.n_x))
        map_names = []
        map_freqs = []

        for i, (band_name, band_map_data) in enumerate(self.map_data.items()):
            map_names.append(band_name)
            map_freqs.append(band_map_data["nom_freq"])

            if "gaussian_filter" in self.map_postprocessing.keys():
                sigma = self.map_postprocessing["gaussian_filter"]["sigma"]

                band_map_numer = sp.ndimage.gaussian_filter(
                    band_map_data["sum"], sigma=sigma
                )
                band_map_denom = sp.ndimage.gaussian_filter(
                    band_map_data["weight"], sigma=sigma
                )

                map_data[i] = band_map_numer / band_map_denom
                map_weight[i] = band_map_denom

            if "median_filter" in self.map_postprocessing.keys():
                map_data[i] = sp.ndimage.median_filter(
                    map_data[i], size=self.map_postprocessing["median_filter"]["size"]
                )

        return Map(
            data=map_data,
            name=map_names,
            weight=map_weight,
            frequency=map_freqs,
            resolution=self.resolution,
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

        band_map_data = {
            "sum": np.zeros((self.n_y, self.n_x)),
            "weight": np.zeros((self.n_y, self.n_x)),
        }

        tods_pbar = tqdm(
            self.tods, desc=f"Running mapper ({band})", disable=not self.verbose
        )  # noqa

        for tod in tods_pbar:
            band_tod = tod.subset(band=band)

            W, D = band_tod.process(**self.tod_postprocessing)
            D *= band_tod.dets.cal.values[..., None]  # convert to KRJ

            dx, dy = band_tod.coords.offsets(frame=self.frame, center=self.center)

            nu = band_tod.dets.band_center.mean()

            del band_tod

            map_sum = sp.stats.binned_statistic_2d(
                dx.ravel(),
                dy.ravel(),
                D.ravel(),
                bins=(self.x_bins, self.y_bins),
                statistic="sum",
            )[0]

            map_weight = sp.stats.binned_statistic_2d(
                dx.ravel(),
                dy.ravel(),
                W.ravel(),
                bins=(self.x_bins, self.y_bins),
                statistic="sum",
            )[0]

            band_map_data["sum"] += map_sum
            band_map_data["weight"] += map_weight

        band_map_data["nom_freq"] = nu

        return band_map_data
