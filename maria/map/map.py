import os
from typing import Tuple

import astropy as ap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import ListedColormap

from ..units import KbrightToJyPix

here, this_filename = os.path.split(__file__)

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/Planck_Parchment_RGB.txt") / 255.0, name="cmb"
)
cmb_cmap.set_bad("white")

mpl.colormaps.register(cmb_cmap)

UNITS_CONFIG = {
    "K_RJ": {"long_name": "Rayleigh-Jeans Temperature [K]"},
    "Jy/pixel": {"long_name": "Jy per pixel"},
}


class Map:
    """
    A map.
    """

    def __init__(
        self,
        name: str = None,
        data: float = np.zeros((1, 1)),
        weight: float = None,
        width: float = None,
        resolution: float = None,
        frequency: float = 100.0,
        center: Tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
    ):
        if units not in UNITS_CONFIG:
            raise ValueError(f"'units' must be one of {list(UNITS_CONFIG.keys())}.")

        self.name = (
            np.atleast_1d(name)
            if name is not None
            else [f"{int(nu)} GHz" for nu in np.atleast_1d(frequency)]
        )
        self.data = data if data.ndim > 2 else data[None]
        self.weight = weight if weight is not None else np.ones(self.data.shape)
        self.center = tuple(np.radians(center)) if degrees else center
        self.frequency = np.atleast_1d(frequency)
        self.frame = frame
        self.units = units

        if not (width is None) ^ (resolution is None):
            raise ValueError("You must pass exactly one of 'width' or 'resolution'.")
        if width is not None:
            if not width > 0:
                raise ValueError("'width' must be positive.")
            width_radians = np.radians(width) if degrees else width
            self.resolution = width_radians / self.n_x
        else:
            if not resolution > 0:
                raise ValueError("'resolution' must be positive.")
            self.resolution = np.radians(resolution) if degrees else resolution

        if len(self.frequency) != self.n_f:
            raise ValueError(
                f"Number of supplied frequencies ({len(self.frequency)}) does not match the "
                f"frequency dimension of the supplied map ({self.n_f})."
            )

        # if self.units == "Jy/pixel":
        # self.to("K_RJ", inplace=True)

        self.header = ap.io.fits.header.Header()

        self.header["CDELT1"] = np.degrees(self.resolution)  # degree
        self.header["CDELT2"] = np.degrees(self.resolution)  # degree
        self.header["CTYPE1"] = "RA---SIN"
        self.header["CUNIT1"] = "deg     "
        self.header["CTYPE2"] = "DEC--SIN"
        self.header["CUNIT2"] = "deg     "
        self.header["CRPIX1"] = self.data.shape[-1]
        self.header["CRPIX2"] = self.data.shape[-2]
        self.header["CRVAL1"] = self.center[0]
        self.header["CRVAL2"] = self.center[1]
        self.header["RADESYS"] = "FK5     "

    def __repr__(self):
        parts = []

        center_degrees = np.degrees(self.center)

        parts.append(f"shape={self.data.shape}")
        parts.append(f"freqs={self.frequency}")
        parts.append(f"center=({center_degrees[0]:.02f}, {center_degrees[1]:.02f})")
        parts.append(f"width={np.degrees(self.width).round()}°")

        return f"Map({', '.join(parts)})"

    @property
    def width(self):
        return self.resolution * self.n_x

    @property
    def height(self):
        return self.resolution * self.n_y

    @property
    def n_f(self):
        return self.data.shape[-3]

    @property
    def n_y(self):
        return self.data.shape[-2]

    @property
    def n_x(self):
        return self.data.shape[-1]

    @property
    def x_side(self):
        x = self.resolution * np.arange(self.n_x)
        return x - x.mean()

    @property
    def y_side(self):
        y = self.resolution * np.arange(self.n_y)
        return y - y.mean()

    def to(self, units, inplace=False):
        data = np.zeros(self.data.shape)

        for i, nu in enumerate(self.frequency):
            if units == self.units:
                data[i] = self.data[i]

            elif units == "K_RJ":
                data[i] = self.data[i] / KbrightToJyPix(
                    nu * 1e9, np.degrees(self.resolution)
                )
            elif units == "Jy/pixel":
                data[i] = self.data[i] * KbrightToJyPix(
                    nu * 1e9, np.degrees(self.resolution)
                )
            else:
                raise ValueError(f"Units '{units}' not implemented.")

            if inplace:
                self.data = data
                self.units = units

        else:
            return Map(
                data=data,
                weight=self.weight,
                resolution=self.resolution,
                frequency=self.frequency,
                center=self.center,
                frame=self.frame,
                degrees=False,
                units=units,
            )

    def plot(
        self, cmap="cmb", rel_vmin=0.001, rel_vmax=0.999, units="degrees", **kwargs
    ):
        for i_freq, freq in enumerate(self.frequency):
            header = fits.header.Header()

            header["RESTFRQ"] = freq

            res_degrees = np.degrees(self.resolution)
            center_degrees = np.degrees(self.center)

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
                self.data[i_freq].T,
                cmap=cmap,
                interpolation="none",
                extent=map_extent,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel(rf"$\Delta\,\theta_x$ [{units}]")
            ax.set_ylabel(rf"$\Delta\,\theta_y$ [{units}]")

            cbar = fig.colorbar(map_im, ax=ax, shrink=1.0)
            cbar.set_label(UNITS_CONFIG[self.units]["long_name"])

    def to_fits(self, filepath):
        self.header = ap.io.fits.header.Header()
        self.header["comment"] = "Made Synthetic observations via maria code"
        self.header["comment"] = "Overwrote resolution and size of the output map"

        self.header["CDELT1"] = np.radians(self.resolution)
        self.header["CDELT2"] = np.radians(self.resolution)

        self.header["CRPIX1"] = self.n_x / 2
        self.header["CRPIX2"] = self.n_y / 2

        self.header["CRVAL1"] = np.radians(self.center[0])
        self.header["CRVAL2"] = np.radians(self.center[1])

        self.header["CTYPE1"] = "RA---SIN"
        self.header["CTYPE2"] = "DEC--SIN"
        self.header["CUNIT1"] = "deg     "
        self.header["CUNIT2"] = "deg     "
        self.header["CTYPE3"] = "FREQ    "
        self.header["CUNIT3"] = "Hz      "
        self.header["CRPIX3"] = 1.000000000000e00

        self.header["comment"] = "Overwrote pointing location of the output map"
        self.header["comment"] = "Overwrote spectral position of the output map"

        if self.units == "Jy/pixel":
            self.to("Jy/pixel")
            self.header["BTYPE"] = "Jy/pixel"

        elif self.units == "K_RJ":
            self.header["BTYPE"] = "Kelvin RJ"

        fits.writeto(
            filename=filepath,
            data=self.data,
            header=self.header,
            overwrite=True,
        )
