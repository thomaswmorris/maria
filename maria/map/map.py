import os
from dataclasses import dataclass, fields
from operator import attrgetter
from typing import List, Tuple

import astropy as ap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import ListedColormap

here, this_filename = os.path.split(__file__)

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/Planck_Parchment_RGB.txt") / 255.0, name="cmb"
)
cmb_cmap.set_bad("white")

mpl.colormaps.register(cmb_cmap)


@dataclass
class Map:
    """
    We define height and width, which determines the shape of the

    This means that there might be non-square pixels
    """

    data: np.array
    frequency: List[float] = 100.0
    center: Tuple[float, float] = ()
    width: float = 1
    height: float = 1
    degrees: bool = True
    inbright: float = 1
    header: ap.io.fits.header.Header = None
    frame: str = "ra_dec"
    units: str = "K_RJ"
    weight: np.array = None

    def __post_init__(self):
        self.frequency = np.atleast_1d(self.frequency)

        if len(self.frequency) != self.data.shape[-3]:
            raise ValueError(
                f"Number of supplied frequencies ({len(self.frequency)}) does not match the "
                f"frequency dimension of the supplied map ({self.data.shape[-2]})."
            )

        self.header = ap.io.fits.header.Header()

        self.header["CDELT1"] = np.degrees(self.x_res)  # degree
        self.header["CDELT2"] = np.degrees(self.y_res)  # degree
        self.header["CTYPE1"] = "RA---SIN"
        self.header["CUNIT1"] = "deg     "
        self.header["CTYPE2"] = "DEC--SIN"
        self.header["CUNIT2"] = "deg     "
        self.header["CRPIX1"] = self.data.shape[-1]
        self.header["CRPIX2"] = self.data.shape[-2]
        self.header["CRVAL1"] = self.center[0]
        self.header["CRVAL2"] = self.center[1]
        self.header["RADESYS"] = "FK5     "

        self.width = self.res * self.n_x
        self.height = self.res * self.n_y

        if self.weight is None:
            self.weight = np.ones(self.data.shape)

    def __repr__(self):
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self))
            for f in fields(self)
            if f.name not in ["data", "weight", "header"]
        )

        nodef_f_repr = ", ".join(f"{name}={value}" for name, value in nodef_f_vals)
        return f"{self.__class__.__name__}({nodef_f_repr})"

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

    def plot(
        self, cmap="cmb", rel_vmin=0.001, rel_vmax=0.999, units="degrees", **kwargs
    ):
        for i_freq, freq in enumerate(self.frequency):
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

            d = self.data.ravel()
            w = self.weight.ravel()

            sort = np.argsort(d)
            d, w = d[sort], w[sort]

            vmin, vmax = np.interp([rel_vmin, rel_vmax], np.cumsum(w) / np.sum(w), d)

            map_im = ax.imshow(
                self.data.T,
                cmap=cmap,
                interpolation="none",
                extent=map_extent,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel(rf"$\Delta\,\theta_x$ [{units}]")
            ax.set_ylabel(rf"$\Delta\,\theta_y$ [{units}]")

            cbar = fig.colorbar(map_im, ax=ax, shrink=1.0)
            cbar.set_label("mJy km/s/pixel")
