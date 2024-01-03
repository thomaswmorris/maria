from dataclasses import dataclass
from typing import List, Tuple

import astropy as ap
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

from .. import utils


@dataclass
class Map:
    """
    We define height and width, which determines the shape of the

    This means that there might be non-square pixels
    """

    freqs: List[float]
    center: Tuple[float, float]
    width: float = 1
    height: float = 1
    degrees: bool = True
    inbright: float = 1
    header: ap.io.fits.header.Header = None
    frame: str = "ra_dec"
    units: str = "K"
    data: np.array = None  # 3D array

    def __post_init__(self):
        assert len(self.freqs) == self.n_freqs

        assert self.data is not None

        self.width = self.res * self.n_x
        self.height = self.res * self.n_y

    @property
    def res(self):
        """
        TODO: don't do this
        """
        return self.x_res

    @property
    def x_res(self):
        return self.width / self.n_x

    @property
    def y_res(self):
        return self.height / self.n_y

    @property
    def n_freqs(self):
        return self.data.shape[0]

    @property
    def n_x(self):
        return self.data.shape[2]

    @property
    def n_y(self):
        return self.data.shape[1]

    @property
    def x_side(self):
        x = self.res * np.arange(self.n_x)
        return x - x.mean()

    @property
    def y_side(self):
        y = self.res * np.arange(self.n_y)
        return y - y.mean()

    def plot(self, cmap="plasma", units="degrees", **kwargs):
        for i_freq, freq in enumerate(self.freqs):
            header = fits.header.Header()

            header["RESTFRQ"] = freq

            res_degrees = self.res if self.degrees else np.degrees(self.res)
            center_degrees = self.center if self.degrees else np.degrees(self.center)

            header["CDELT1"] = res_degrees  # degree
            header["CDELT2"] = res_degrees  # degree

            header["CRPIX1"] = self.n_x / 2
            header["CRPIX2"] = self.n_y / 2

            header["CTYPE1"] = "RA---SIN"
            header["CUNIT1"] = "deg     "
            header["CTYPE2"] = "DEC--SIN"
            header["CUNIT2"] = "deg     "
            header["RADESYS"] = "FK5     "

            header["CRVAL1"] = center_degrees[0]
            header["CRVAL2"] = center_degrees[1]
            wcs_input = WCS(header, naxis=2)  # noqa F401

            fig = plt.figure()

            ax = fig.add_subplot(1, 1, 1)  # , projection=wcs_input)

            ax.set_title(f"{freq} GHz")

            map_extent_radians = [
                -self.width / 2,
                self.width / 2,
                -self.height / 2,
                self.height / 2,
            ]
            if self.degrees:
                map_extent_radians = np.radians(map_extent_radians)

            if units == "degrees":
                map_extent = np.degrees(map_extent_radians)
            if units == "arcmin":
                map_extent = 60 * np.degrees(map_extent_radians)
            if units == "arcsec":
                map_extent = 3600 * np.degrees(map_extent_radians)

            map_im = ax.imshow(
                self.data.T, cmap=cmap, interpolation="none", extent=map_extent
            )

            ax.set_xlabel(rf"$\Delta\,\theta_x$ [{units}]")
            ax.set_ylabel(rf"$\Delta\,\theta_y$ [{units}]")

            cbar = fig.colorbar(map_im, ax=ax, shrink=1.0)
            cbar.set_label("mJy km/s/pixel")


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _initialize_map(self):
        if not self.map_file:
            return

        self.input_map_file = self.map_file
        hudl = ap.io.fits.open(self.map_file)

        map_data = hudl[0].data
        if map_data.ndim < 2:
            raise ValueError()
        elif map_data.ndim == 2:
            map_data = map_data[None]

        map_n_freqs, map_n_y, map_n_x = map_data.shape

        if map_n_freqs != len(self.map_freqs):
            raise ValueError()

        map_width = self.map_res * map_n_x
        map_height = self.map_res * map_n_y

        self.raw_map_data = map_data.copy()

        res_degrees = self.map_res if self.degrees else np.degrees(self.map_res)

        if self.map_units == "Jy/pixel":
            for i, nu in enumerate(self.map_freqs):
                map_data[i] = map_data[i] / utils.units.KbrightToJyPix(
                    1e9 * nu, res_degrees, res_degrees
                )

        self.map_data = map_data

        self.input_map = Map(
            data=map_data,
            header=hudl[0].header,
            freqs=np.atleast_1d(self.map_freqs),
            width=np.radians(map_width) if self.degrees else map_width,
            height=np.radians(map_height) if self.degrees else map_height,
            center=np.radians(self.map_center) if self.degrees else map_height,
            degrees=False,
            frame=self.pointing_frame,
            inbright=self.map_inbright,
            units=self.map_units,
        )

        self.input_map.header["HISTORY"] = "History_input_adjustments"
        self.input_map.header["comment"] = "Changed input CDELT1 and CDELT2"
        self.input_map.header["comment"] = (
            "Changed surface brightness units to " + self.input_map.units
        )
        self.input_map.header["comment"] = "Repositioned the map on the sky"

        if self.input_map.inbright is not None:
            self.input_map.data *= self.input_map.inbright / np.nanmax(
                self.input_map.data
            )
            self.input_map.header["comment"] = "Amplitude is rescaled."

    def _run(self, **kwargs):
        self.sample_maps()

    def _sample_maps(self):
        dx, dy = self.det_coords.offsets(
            frame=self.map_frame, center=self.input_map.center
        )

        self.data["map"] = np.zeros((dx.shape))

        freqs = tqdm(self.input_map.freqs) if self.verbose else self.input_map.freqs
        for i, nu in enumerate(freqs):
            if self.verbose:
                freqs.set_description(f"Sampling map at {nu} GHz")

            # nu is in GHz, f is in Hz
            nu_fwhm = utils.beam.angular_fwhm(
                fwhm_0=self.array.primary_size, z=np.inf, f=1e9 * nu
            )
            nu_map_filter = utils.beam.make_beam_filter(
                fwhm=nu_fwhm, res=self.input_map.res
            )
            filtered_nu_map_data = utils.beam.separably_filter(
                self.input_map.data[i], nu_map_filter
            )

            # band_res_radians = 1.22 * (299792458 / (1e9 * nu)) / self.array.primary_size
            # band_res_pixels = band_res_radians / self.input_map.res
            # FWHM_TO_SIGMA = 2.355
            # band_beam_sigma_pixels = band_res_pixels / FWHM_TO_SIGMA

            # # band_beam_filter = self.array.

            # # filtered_map_data = sp.ndimage.convolve()
            det_freq_response = self.array.passbands(nu=np.array([nu]))[:, 0]
            det_mask = det_freq_response > -np.inf  # -1e-3

            samples = sp.interpolate.RegularGridInterpolator(
                (self.input_map.x_side, self.input_map.y_side),
                filtered_nu_map_data,
                bounds_error=False,
                fill_value=0,
                method="linear",
            )((dx[det_mask], dy[det_mask]))

            self.data["map"][det_mask] = samples
