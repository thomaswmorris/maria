from dataclasses import dataclass
from typing import List, Tuple

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

        # self.res = np.radians(self.res) if self.degrees else self.res
        # self.center = np.radians(self.center) if self.degrees else self.center

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
