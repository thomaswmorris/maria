import os

import astropy as ap
import numpy as np
import scipy as sp
from astropy.io import fits  # noqa F401

from . import utils
from .map import Map

here, this_filename = os.path.split(__file__)

MAP_CONFIGS = utils.io.read_yaml(f"{here}/configs/sky.yml")
MAP_PARAMS = set()
for key, config in MAP_CONFIGS.items():
    MAP_PARAMS |= set(config.keys())


class InvalidNBandsError(Exception):
    def __init__(self, invalid_nbands):
        super().__init__(
            f"Number of bands  '{invalid_nbands}' don't match the cube size."
            f"The input fits file must be an image or a cube that match the number of bands"
        )


class MapMixin:
    """
    This simulates scanning over celestial sources.
    """

    def _initialize_map(self):
        if not self.map_file:
            return

        self.input_map_file = self.map_file
        hudl = ap.io.fits.open(self.map_file)

        freqs = np.unique(self.array.dets.band_center)

        self.input_map = Map(
            data=hudl[0].data[None],
            header=hudl[0].header,
            freqs=np.atleast_1d(freqs),
            width=np.radians(self.map_width),
            height=np.radians(self.map_height),
            center=np.radians(self.map_center),
            frame=self.map_frame,
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

        if self.input_map.units == "Jy/pixel":
            for i, nu in enumerate(self.input_map.freqs):
                self.input_map.data[i] = self.input_map.data[
                    i
                ] / utils.units.KbrightToJyPix(
                    1e9 * nu,
                    np.rad2deg(self.input_map.res),
                    np.rad2deg(self.input_map.res),
                )

    def _run(self, **kwargs):
        self.sample_maps()

    def _sample_maps(self):
        dx, dy = self.det_coords.offsets(
            frame=self.map_frame, center=self.input_map.center
        )

        self.data["map"] = np.zeros((dx.shape))

        for i, nu in enumerate(self.input_map.freqs):
            det_freq_response = self.array.passbands(nu=np.array([nu]))[:, 0]

            det_mask = det_freq_response > -1e-3

            samples = sp.interpolate.RegularGridInterpolator(
                (self.input_map.x_side, self.input_map.x_side),
                self.input_map.data[i],
                bounds_error=False,
                fill_value=0,
                method="linear",
            )((dx[det_mask], dy[det_mask]))

            self.data["map"][det_mask] = samples
