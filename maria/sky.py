import os
from dataclasses import dataclass

import astropy as ap
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits  # noqa F401

from . import base, utils

here, this_filename = os.path.split(__file__)

MAP_CONFIGS = utils.io.read_yaml(f"{here}/configs/sky.yml")
MAP_PARAMS = set()
for key, config in MAP_CONFIGS.items():
    MAP_PARAMS |= set(config.keys())


@dataclass
class Map:
    data: np.array  # 3D array
    freqs: np.array
    res: float
    inbright: float
    center: tuple
    header: ap.io.fits.header.Header = None
    frame: str = "ra_dec"
    units: str = "K"

    @property
    def n_freqs(self):
        return len(self.freqs)

    @property
    def shape(self):
        return self.data.shape[-2:]

    @property
    def n_x(self):
        return self.shape[0]

    @property
    def n_y(self):
        return self.shape[1]

    @property
    def x_side(self):
        x = self.res * np.arange(self.n_x)
        return x - x.mean()

    @property
    def y_side(self):
        y = self.res * np.arange(self.n_y)
        return y - y.mean()

    # @property
    # def x_side(self):
    #     return self.x_side + self.center[0]

    # @property
    # def y_side(self):
    #     return self.y_side + self.center[1]

    @property
    def X_Y(self):
        return np.meshgrid(self.x_side, self.y_side)

    # @property
    # def rel_X_Y(self):
    #     return np.meshgrid(self.x_side, self.y_side)

    def plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=128)
        map_extent = np.degrees(
            [self.x_side.min(), self.x_side.max(), self.y_side.min(), self.y_side.max()]
        )
        map_im = ax.imshow(self.data[0], extent=map_extent)
        if self.frame == "ra_dec":
            ax.set_xlabel("RA (deg.)")
            ax.set_ylabel("dec. (deg.)")
        clb = fig.colorbar(mappable=map_im, shrink=0.8, aspect=32)
        clb.set_label(self.units)


class InvalidNBandsError(Exception):
    def __init__(self, invalid_nbands):
        super().__init__(
            f"Number of bands  '{invalid_nbands}' don't match the cube size."
            f"The input fits file must be an image or a cube that match the number of bands"
        )


class BaseSkySimulation(base.BaseSimulation):
    """
    This simulates scanning over celestial sources.
    """

    def __init__(self, array, pointing, site, **kwargs):
        super().__init__(array, pointing, site, **kwargs)

        AZIM, ELEV = utils.coords.xy_to_lonlat(
            self.array.offset_x[:, None],
            self.array.offset_y[:, None],
            self.pointing.az,
            self.pointing.el,
        )

        self.RA, self.DEC = self.coordinator.transform(
            self.pointing.time,
            AZIM,
            ELEV,
            in_frame="az_el",
            out_frame="ra_dec",
        )

        self.X, self.Y = utils.coords.lonlat_to_xy(
            self.RA, self.DEC, self.RA.mean(), self.DEC.mean()
        )


class MapSimulation(BaseSkySimulation):
    """
    This simulates scanning over celestial sources.
    """

    def __init__(self, array, pointing, site, map_file, **kwargs):
        super().__init__(array, pointing, site, **kwargs)

        self.input_map_file = map_file
        hudl = ap.io.fits.open(map_file)

        freqs = np.unique(self.array.dets.band_center)

        self.input_map = Map(
            data=hudl[0].data[None],
            header=hudl[0].header,
            freqs=np.atleast_1d(kwargs.get("map_freqs", freqs)),
            res=np.radians(kwargs.get("map_res", 1 / 1000)),
            center=np.radians(kwargs.get("map_center", (10.5, 4))),
            frame=kwargs.get("map_frame", "ra_dec"),
            inbright=kwargs.get("map_inbright", None),
            units=kwargs.get("map_units", "K"),
        )

        self.map_X, self.map_Y = utils.coords.lonlat_to_xy(
            self.RA, self.DEC, *self.input_map.center
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
        self.data = self.map_samples

    def sample_maps(self):
        self.map_samples = np.zeros((self.RA.shape))

        for i, nu in enumerate(self.input_map.freqs):
            det_freq_response = self.array.passbands(nu=np.array([nu]))[:, 0]

            det_mask = det_freq_response > -1e-3

            samples = sp.interpolate.RegularGridInterpolator(
                (self.input_map.x_side, self.input_map.x_side),
                self.input_map.data[i],
                bounds_error=False,
                fill_value=0,
                method="linear",
            )((self.map_X[det_mask], self.map_Y[det_mask]))

            self.map_samples[det_mask] = samples
