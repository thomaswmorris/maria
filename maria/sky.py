import os

import astropy as ap
import numpy as np
import scipy as sp
from astropy.io import fits  # noqa F401

from . import utils
from .map import Map

here, this_filename = os.path.split(__file__)


class InvalidNBandsError(Exception):
    def __init__(self, invalid_nbands):
        super().__init__(
            f"Number of bands '{invalid_nbands}' don't match the cube size."
            f"The input fits file must be an image or a cube that match the number of bands"
        )


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

        for i, nu in enumerate(self.input_map.freqs):
            band_res_radians = 1.22 * (299792458 / (1e9 * nu)) / self.array.primary_size
            band_res_pixels = band_res_radians / self.input_map.res
            FWHM_TO_SIGMA = 2.355
            band_beam_sigma_pixels = band_res_pixels / FWHM_TO_SIGMA

            band_map_data = sp.ndimage.gaussian_filter(
                self.input_map.data[i],
                sigma=(band_beam_sigma_pixels, band_beam_sigma_pixels),
            )

            det_freq_response = self.array.passbands(nu=np.array([nu]))[:, 0]
            det_mask = det_freq_response > -np.inf  # -1e-3

            samples = sp.interpolate.RegularGridInterpolator(
                (self.input_map.x_side, self.input_map.y_side),
                band_map_data,
                bounds_error=False,
                fill_value=0,
                method="linear",
            )((dx[det_mask], dy[det_mask]))

            self.data["map"][det_mask] = samples
