import os
from typing import Tuple

import healpy as hp
import numpy as np
import scipy as sp
from astropy.io import fits

from .. import utils
from . import Map

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)

MAPPER_CONFIGS = utils.io.read_yaml(f"{here}/../configs/mappers.yml")
MAPPERS = list((MAPPER_CONFIGS.keys()))


class InvalidMapperError(Exception):
    def __init__(self, invalid_mapper):
        print(
            f"The mapper '{invalid_mapper}' is not in the database of default mappers."
            f"Default mappers are: {MAPPERS}"
        )


class BaseMapper:
    """
    The base class for mapping.
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0, 0),
        frame: str = "ra_dec",
        width: float = 5,
        height: float = 5,
        res: float = 1 / 60,
        filter_tods: bool = True,
        smoothing: float = 8,
        **kwargs,
    ):
        self.res = res
        self.frame = frame
        self.center = np.radians(center)
        self.width = np.radians(width)
        self.height = np.radians(height)
        self.filter_tods = filter_tods
        self.smoothing = smoothing

        self.n_x = int(np.maximum(1, self.width / self.res))
        self.n_y = int(np.maximum(1, self.height / self.res))

        self.x_bins = np.linspace(-0.5 * self.width, 0.5 * self.width, self.n_x + 1)
        self.y_bins = np.linspace(-0.5 * self.height, 0.5 * self.height, self.n_y + 1)

        self.tods = []

    def plot(self):
        self.map.plot()

    @property
    def n_maps(self):
        return len(self.maps)

    # @property
    # def maps(self):
    #     return {
    #         key: self.map_sums[key]
    #         / np.where(self.map_cnts[key], self.map_cnts[key], np.nan)
    #         for key in self.map_sums.keys()
    #     }

    def smoothed_maps(self, smoothing=1):
        smoothed_maps = {}

        for key in self.map_sums.keys():
            SUMS = sp.ndimage.gaussian_filter(
                self.map_sums[key], sigma=(smoothing, smoothing)
            )
            CNTS = sp.ndimage.gaussian_filter(
                self.map_cnts[key], sigma=(smoothing, smoothing)
            )

            smoothed_maps[key] = SUMS / CNTS

        return smoothed_maps

    def add_tods(self, tods):
        for tod in np.atleast_1d(tods):
            self.tods.append(tod)

    @property
    def get_map_center_lonlat(self):
        for tod in self.tods:
            mean_unit_vec = hp.ang2vec(
                np.pi / 2 - tod.LAT.ravel(), tod.LON.ravel()
            ).mean(axis=0)
            mean_unit_vec /= np.sqrt(np.sum(np.square(mean_unit_vec)))
            mean_unit_colat, mean_unit_lon = np.r_[hp.vec2ang(mean_unit_vec)]

        return mean_unit_lon, np.pi / 2 - mean_unit_colat

    def save_maps(self, filepath):
        self.header = self.tods[0].header
        self.header["comment"] = "Made Synthetic observations via maria code"
        self.header["comment"] = "Overwrote resolution and size of the output map"

        self.header["CDELT1"] = np.rad2deg(self.map_res)
        self.header["CDELT2"] = np.rad2deg(self.map_res)

        self.header["CRPIX1"] = self.maps[list(self.maps.keys())[0]].shape[0] / 2
        self.header["CRPIX2"] = self.maps[list(self.maps.keys())[0]].shape[1] / 2

        self.header["CRVAL1"] = np.rad2deg(self.tods[0].cntr[0])
        self.header["CRVAL2"] = np.rad2deg(self.tods[0].cntr[1])

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
        if self.tods[0].unit == "Jy/pixel":
            self.header["BUNIT"] = "Jy/pixel "
        else:
            self.header["BUNIT"] = "Kelvin RJ"

        save_maps = np.zeros(
            (
                len(self.maps.keys()),
                self.maps[list(self.maps.keys())[0]].shape[0],
                self.maps[list(self.maps.keys())[0]].shape[1],
            )
        )
        for i, key in enumerate(self.maps.keys()):
            # what is this? --> Frequency information in the header
            self.header["CRVAL3"] = self.nom_freqs[key] * 1e9
            self.header["CDELT3"] = self.nom_freqwidth[key] * 1e9

            # save_maps[i] = self.maps[list(self.maps.keys())[i]]
            sigma_smooth = self.map_smt / 3600 / np.rad2deg(self.map_res) / 2.355
            save_maps[i] = self.smoothed_maps(sigma_smooth)[list(self.maps.keys())[i]]

            if self.tods[0].unit == "Jy/pixel":
                save_maps[i] *= utils.units.KbrightToJyPix(
                    self.header["CRVAL3"], self.header["CDELT1"], self.header["CDELT2"]
                )

        fits.writeto(
            filename=filepath,
            data=save_maps,
            header=self.header,
            overwrite=True,
        )


class BinMapper(BaseMapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._nmtr = kwargs.get("n_modes_to_remove", 0)

    def _fourier_filter(self, tod_dat, tod_time):
        ffilt = [0.08, 51.0]  # high-pass and low-pass filters, in Hz
        width = 0.05

        n = len(tod_time)
        dt = tod_time[1] - tod_time[0]
        freqs = np.fft.fftfreq(n, dt)

        ndet, nint = tod_dat.shape
        tfft = np.fft.fft(tod_dat)

        lpf = np.ones(n)
        hpf = np.ones(n)
        if ffilt[1] != 0:
            lpf = self._lpcos_filter(
                freqs, [ffilt[1] * (1 - width), ffilt[1] * (1 + width)]
            )
        if ffilt[0] != 0:
            hpf = self._hpcos_filter(
                freqs, [ffilt[0] * (1 - width), ffilt[0] * (1 + width)]
            )

        filt = np.outer(np.ones(ndet), hpf * lpf)
        filttod = np.real(np.fft.ifft(tfft * filt))
        return filttod

    def _lpcos_filter(self, k, par):
        k1 = par[0]
        k2 = par[1]
        filter = k * 0.0
        filter[k < k1] = 1.0
        filter[k >= k1] = 0.5 * (1 + np.cos(np.pi * (k[k >= k1] - k1) / (k2 - k1)))
        filter[k > k2] = 0.0
        return filter

    def _hpcos_filter(self, k, par):
        k1 = par[0]
        k2 = par[1]
        filter = k * 0.0
        filter[k < k1] = 0.0
        filter[k >= k1] = 0.5 * (1 - np.cos(np.pi * (k[k >= k1] - k1) / (k2 - k1)))
        filter[k > k2] = 1.0
        return filter

    def run(self):
        self.ubands = sorted(
            [band for tod in self.tods for band in np.unique(tod.dets.band)]
        )

        self.band_data = {}

        # self.nom_freqwidth = []
        self.map_sums = {band: np.zeros((self.n_x, self.n_y)) for band in self.ubands}
        self.map_cnts = {band: np.zeros((self.n_x, self.n_y)) for band in self.ubands}

        map_data = np.zeros((len(self.ubands), self.n_x, self.n_y))

        for iband, band in enumerate(np.unique(self.ubands)):
            self.band_data[band] = {}

            for tod in self.tods:
                # compute detector offsets WRT the map
                dx, dy = tod.coords.offsets(frame=self.frame, center=self.center)

                band_mask = tod.dets.band == band

                # filter, if needed
                if self.filter_tods:
                    tod.data[band_mask] = self._fourier_filter(
                        tod.data[band_mask], tod.time
                    )

                if self._nmtr > 0:
                    u, s, v = np.linalg.svd(
                        sp.signal.detrend(tod.data[band_mask]), full_matrices=False
                    )
                    DATA = (
                        u[:, self._nmtr :] @ np.diag(s[self._nmtr :]) @ v[self._nmtr :]
                    )
                else:
                    DATA = sp.signal.detrend(tod.data[band_mask])

                map_sum = sp.stats.binned_statistic_2d(
                    dx[band_mask].ravel(),
                    dy[band_mask].ravel(),
                    DATA.ravel(),
                    bins=(self.x_bins, self.y_bins),
                    statistic="sum",
                )[0]

                map_cnt = sp.stats.binned_statistic_2d(
                    dx[band_mask].ravel(),
                    dy[band_mask].ravel(),
                    DATA.ravel(),
                    bins=(self.x_bins, self.y_bins),
                    statistic="count",
                )[0]

                self.map_sums[band] += map_sum
                self.map_cnts[band] += map_cnt

            self.band_data[band]["nom_freq"] = tod.dets.loc[
                band_mask, "band_center"
            ].mean()
            # self.nom_freqwidth[band] = tod.dets.loc["band_width.mean()

            mask = self.map_cnts[band] > 0
            map_data[iband] = np.where(mask, self.map_sums[band], np.nan) / np.where(
                mask, self.map_cnts[band], 1
            )

        self.map = Map(
            data=map_data,
            freqs=np.array([self.band_data[band]["nom_freq"] for band in self.ubands]),
            center=self.center,
        )
