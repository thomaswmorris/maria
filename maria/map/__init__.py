from dataclasses import dataclass
from typing import Tuple

import astropy as ap
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


@dataclass
class Map:
    """
    We define height and width, which determines the shape of the

    This means that there might be non-square pixels
    """

    data: np.array  # 3D array
    freqs: np.array
    center: Tuple[float, float]
    height: float = np.radians(1)
    width: float = np.radians(1)
    inbright: float = 1
    header: ap.io.fits.header.Header = None
    frame: str = "ra_dec"
    units: str = "K"

    def __post_init__(self):
        ...  # X, Y = np.meshgrid(self.x_side, self.y_side)

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

    @property
    def X_Y(self):
        return np.meshgrid(self.x_side, self.y_side)

    def plot(self, cmap="plasma", **kwargs):
        for i_freq, freq in enumerate(self.freqs):
            header = fits.header.Header()

            header["RESTFRQ"] = freq

            header["CDELT1"] = self.res  # degree
            header["CDELT2"] = self.res  # degree

            header["CRPIX1"] = self.n_x / 2
            header["CRPIX2"] = self.n_y / 2

            header["CTYPE1"] = "RA---SIN"
            header["CUNIT1"] = "deg     "
            header["CTYPE2"] = "DEC--SIN"
            header["CUNIT2"] = "deg     "
            header["RADESYS"] = "FK5     "

            header["CRVAL1"] = np.degrees(self.center[0])
            header["CRVAL2"] = np.degrees(self.center[1])
            wcs_input = WCS(header, naxis=2)

            fig = plt.figure()

            ax = fig.add_subplot(1, 1, 1, projection=wcs_input)

            ax.set_title(f"{freq} GHz")

            map_im = ax.imshow(self.data.T, cmap=cmap, interpolation="none")

            cbar = fig.colorbar(map_im, ax=ax, shrink=1.0)
            cbar.set_label("mJy km/s/pixel")

            ra, dec = ax.coords
            ra.set_major_formatter("hh:mm:ss")
            dec.set_major_formatter("dd:mm:ss")
            ra.set_axislabel(r"RA [J2000]", size=11)
            dec.set_axislabel(r"Dec [J2000]", size=11)
            ra.set_separator(("h", "m"))

            # pixel_sky_l = wcs_input.world_to_pixel(sky_l)
            # pixel_sky_u = wcs_input.world_to_pixel(sky_u)
            # ax.axis(
            #     xmax=pixel_sky_l[0], ymin=pixel_sky_l[1], xmin=pixel_sky_u[0], ymax=pixel_sky_u[1]
            # )
            # plt.show()
