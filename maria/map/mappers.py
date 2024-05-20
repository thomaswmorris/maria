import os
from typing import Sequence, Tuple

import numpy as np
import scipy as sp
from astropy.io import fits
from todder import TOD

from .. import units, utils
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
        res: float = 0.01,
        frame: str = "ra_dec",
        units: str = "K_RJ",
        degrees: bool = True,
        calibrate: bool = False,
        tods: Sequence[TOD] = [],
    ):
        self.res = np.radians(res) if degrees else res
        self.center = np.radians(center) if degrees else center
        self.width = np.radians(width) if degrees else width
        self.height = np.radians(height) if degrees else height
        self.degrees = degrees
        self.calibrate = calibrate
        self.frame = frame
        self.units = units

        self.n_x = int(np.maximum(1, self.width / self.res))
        self.n_y = int(np.maximum(1, self.height / self.res))

        self.x_bins = np.linspace(-0.5 * self.width, 0.5 * self.width, self.n_x + 1)
        self.y_bins = np.linspace(-0.5 * self.height, 0.5 * self.height, self.n_y + 1)

        self.tods = list(np.atleast_1d(tods))

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

    def save_maps(self, filepath):
        self.header = self.tods[0].header
        self.header["comment"] = "Made Synthetic observations via maria code"
        self.header["comment"] = "Overwrote resolution and size of the output map"

        self.header["CDELT1"] = np.rad2deg(self.res)
        self.header["CDELT2"] = np.rad2deg(self.res)

        self.header["CRPIX1"] = self.n_x / 2
        self.header["CRPIX2"] = self.n_y / 2

        self.header["CRVAL1"] = np.rad2deg(self.center[0])
        self.header["CRVAL2"] = np.rad2deg(self.center[1])

        self.header["CTYPE1"] = "RA---SIN"
        self.header["CTYPE2"] = "DEC--SIN"
        self.header["CUNIT1"] = "deg     "
        self.header["CUNIT2"] = "deg     "
        self.header["CTYPE3"] = "FREQ    "
        self.header["CUNIT3"] = "Hz      "
        self.header["CRPIX3"] = 1.000000000000e00

        self.header["comment"] = "Overwrote pointing location of the output map"
        self.header["comment"] = "Overwrote spectral position of the output map"

        self.header["BTYPE"] = "Intensity"

        if self.map.units == "Jy/pixel":
            self.header["BUNIT"] = "Jy/pixel "
        elif self.map.units == "K_RJ":
            self.header["BUNIT"] = "Kelvin RJ"  # tods are always in Kelvin
        else:
            raise ValueError(f"Units {self.map.units} not implemented.")

        save_maps = np.zeros((len(self.map.frequency), self.n_x, self.n_y))

        for i, key in enumerate(self.band_data.keys()):
            self.header["CRVAL3"] = self.band_data[key]["band_center"] * 1e9
            self.header["CDELT3"] = self.band_data[key]["band_width"] * 1e9

            save_maps[i] = self.map.data[i]

            if self.map.units == "Jy/pixel":
                save_maps[i] *= units.KbrightToJyPix(
                    self.header["CRVAL3"], self.header["CDELT1"], self.header["CDELT2"]
                )

        fits.writeto(
            filename=filepath,
            data=save_maps,
            header=self.header,
            overwrite=True,
        )


class BinMapper(BaseMapper):
    def __init__(
        self,
        center: Tuple[float, float] = (0, 0),
        width: float = 1,
        height: float = 1,
        res: float = 0.01,
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
            res=res,
            frame=frame,
            degrees=degrees,
            calibrate=calibrate,
            tods=tods,
            units=units,
        )

        self.tod_postprocessing = tod_postprocessing
        self.map_postprocessing = map_postprocessing

    def run(self):
        self.band = sorted(
            np.unique(
                [band for tod in self.tods for band in np.unique(tod.dets.band_name)]
            )
        )

        self.band_data = {}

        self.raw_map_sums = {band: np.zeros((self.n_x, self.n_y)) for band in self.band}
        self.raw_map_cnts = {band: np.zeros((self.n_x, self.n_y)) for band in self.band}

        self.map_data = np.zeros((len(self.band), self.n_y, self.n_x))
        self.map_weight = np.zeros((len(self.band), self.n_y, self.n_x))

        for iband, band in enumerate(np.unique(self.band)):
            self.band_data[band] = {}

            for tod in self.tods:
                # compute detector offsets WRT the map
                dx, dy = tod.coords.offsets(frame=self.frame, center=self.center)

                band_mask = tod.dets.band_name == band

                if self.calibrate:
                    D = tod.data_calibrated.copy()[band_mask]
                else:
                    D = tod.data.copy()[band_mask]

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
                        dk=self.tod_postprocessing["despline"].get("knot_spacing", 10),
                        order=self.tod_postprocessing["despline"].get(
                            "spline_order", 3
                        ),
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

                self.raw_map_sums[band] += map_sum
                self.raw_map_cnts[band] += map_cnt

            self.band_data[band]["band_center"] = tod.dets.band_center.mean()
            self.band_data[band]["band_width"] = 30

            band_map_numer = self.raw_map_sums[band].copy()
            band_map_denom = self.raw_map_cnts[band].copy()

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

            self.map_data[iband] = np.where(mask, band_map_numer, np.nan) / np.where(
                mask, band_map_denom, 1
            )

            self.map_weight[iband] = band_map_denom

        self.map = Map(
            data=self.map_data,
            weight=self.map_weight,
            frequency=np.array(
                [self.band_data[band]["band_center"] for band in self.band]
            ),
            width=np.degrees(self.width) if self.degrees else self.width,
            height=np.degrees(self.height) if self.degrees else self.height,
            center=np.degrees(self.center) if self.degrees else self.center,
            degrees=self.degrees,
            units=self.units,
        )
