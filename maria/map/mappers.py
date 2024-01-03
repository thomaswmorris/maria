import os
from typing import Tuple

import numpy as np
import scipy as sp
from astropy.io import fits

from .. import todder
from . import Map

np.seterr(invalid="ignore")

here, this_filename = os.path.split(__file__)


class BaseMapper:
    """
    The base class for mapping.

    Units
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0, 0),
        width: float = 1,
        height: float = 1,
        res: float = 0.01,
        frame: str = "ra_dec",
        filter_tods: bool = True,
        smoothing: float = 8,
        degrees: bool = True,
        ffilter: float = 0.08,
        **kwargs,
    ):
        self.res = np.radians(res) if degrees else res
        self.center = np.radians(center) if degrees else center
        self.width = np.radians(width) if degrees else width
        self.height = np.radians(height) if degrees else height
        self.degrees = degrees
        self.frame = frame
        self.filter_tods = filter_tods
        self.smoothing = smoothing
        self.ffilter = ffilter

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

        # if self.tods[0].unit == "Jy/pixel":
        #     self.header["BUNIT"] = "Jy/pixel "
        # else:
        self.header["BUNIT"] = "Kelvin RJ"  # tods are always in Kelvin

        save_maps = np.zeros((len(self.map.freqs), self.n_x, self.n_y))

        for i, key in enumerate(self.band_data.keys()):
            # what is this? --> Frequency information in the header
            self.header["CRVAL3"] = self.band_data[key]["nom_freq"] * 1e9
            self.header["CDELT3"] = self.band_data[key]["nom_freqwidth"] * 1e9

            sigma_smooth = self.smoothing / 3600 / np.rad2deg(self.res) / 2.355
            save_maps[i] = self.smoothed_maps(sigma_smooth)[
                list(self.band_data.keys())[i]
            ]

            # if self.tods[0].unit == "Jy/pixel":
            #     save_maps[i] *= utils.units.KbrightToJyPix(
            #         self.header["CRVAL3"], self.header["CDELT1"], self.header["CDELT2"]
            #     )

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
        ffilt = [self.ffilter, 51.0]  # high-pass and low-pass filters, in Hz
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

        self.map_sums = {band: np.zeros((self.n_x, self.n_y)) for band in self.ubands}
        self.map_cnts = {band: np.zeros((self.n_x, self.n_y)) for band in self.ubands}

        self.map_data = np.zeros((len(self.ubands), self.n_x, self.n_y))

        for iband, band in enumerate(np.unique(self.ubands)):
            self.band_data[band] = {}

            for tod in self.tods:
                # compute detector offsets WRT the map
                dx, dy = tod.coords.offsets(frame=self.frame, center=self.center)

                band_mask = tod.dets.band == band

                window = sp.signal.windows.tukey(tod.nt, alpha=0.1)

                # raw_band_data = sp.signal.detrend(tod.data[band_mask])
                d = sp.signal.detrend(tod.data[band_mask].copy()) * window
                w = np.ones(tod.nd)[:, None] * window

                if self._nmtr > 0:
                    U, V = todder.utils.decompose(
                        d, downsample_rate=np.maximum(tod.fs, 1)
                    )
                    d = U[:, self._nmtr :] @ V[self._nmtr :]

                # filter, if needed
                if self.filter_tods:
                    d = self._fourier_filter(d, tod.time)

                map_sum = sp.stats.binned_statistic_2d(
                    dx[band_mask].ravel(),
                    dy[band_mask].ravel(),
                    d.ravel(),
                    bins=(self.x_bins, self.y_bins),
                    statistic="sum",
                )[0]

                map_cnt = sp.stats.binned_statistic_2d(
                    dx[band_mask].ravel(),
                    dy[band_mask].ravel(),
                    w.ravel(),
                    bins=(self.x_bins, self.y_bins),
                    statistic="sum",
                )[0]

                self.DATA = d

                self.map_sums[band] += map_sum
                self.map_cnts[band] += map_cnt

            self.band_data[band]["nom_freq"] = tod.dets.loc[
                band_mask, "band_center"
            ].mean()

            self.band_data[band]["nom_freqwidth"] = tod.dets.loc[
                band_mask, "band_width"
            ].mean()

            mask = self.map_cnts[band] > 0
            self.map_data[iband] = np.where(
                mask, self.map_sums[band], np.nan
            ) / np.where(mask, self.map_cnts[band], 1)

        self.map = Map(
            data=self.map_data,
            freqs=np.array([self.band_data[band]["nom_freq"] for band in self.ubands]),
            width=np.degrees(self.width) if self.degrees else self.width,
            height=np.degrees(self.height) if self.degrees else self.height,
            center=np.degrees(self.center) if self.degrees else self.center,
            degrees=self.degrees,
        )
