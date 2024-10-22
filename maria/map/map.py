import os
from typing import Tuple

import astropy as ap
import matplotlib as mpl
import numpy as np
from astropy.io import fits
from matplotlib.colors import ListedColormap

from ..plotting import plot_map
from ..units import MAP_UNITS, Angle, KbrightToJyPix

here, this_filename = os.path.split(__file__)

# from https://gist.github.com/zonca/6515744
cmb_cmap = ListedColormap(
    np.loadtxt(f"{here}/Planck_Parchment_RGB.txt") / 255.0, name="cmb"
)
cmb_cmap.set_bad("white")

mpl.colormaps.register(cmb_cmap)


class Map:
    """
    A map, with shape (n_time, n_nu, n_x, n_y).
    """

    def __init__(
        self,
        data: float,
        nu: float = 100.0,
        time: float = None,
        weight: float = None,
        width: float = None,
        resolution: float = None,
        center: Tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
    ):
        if units not in MAP_UNITS:
            raise ValueError(f"'units' must be one of {list(MAP_UNITS.keys())}.")

        self.nu = np.array([np.nan]) if nu is None else np.atleast_1d(nu)
        self.time = np.array([np.nan]) if time is None else np.atleast_1d(time)

        # give it four dimensions
        self.data = data * np.ones((1, 1, 1, 1))

        if len(self.time) != self.data.shape[0]:
            raise ValueError(
                f"time has length {len(self.time)} but map has shape (t, nu, x, y) = {data.shape}."
            )
        if len(self.nu) != self.data.shape[1]:
            raise ValueError(
                f"nu has length {len(self.nu)} but map has shape (t, nu, x, y) = {data.shape}."
            )

        # if time is not None or data.ndim == 4:
        #     if self.data.ndim != 4:
        #         raise ValueError("Map data must be 4-dimensional (time, nuuency, x, y).")
        #     if self.data.shape[0] != len(time):
        #         raise ValueError(f"Time has shape {time.shape} but map has shape {data.shape}.")
        #     self.time = time
        # else:
        #     self.time = np.array([ttime.time()])

        self.weight = weight if weight is not None else np.ones(self.data.shape)
        self.center = tuple(np.radians(center)) if degrees else center

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

        if len(self.nu) != self.n_f:
            raise ValueError(
                f"Number of supplied nuuencies ({len(self.nu)}) does not match the "
                f"nu dimension of the supplied map ({self.n_f})."
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
        parts.append(f"nus={self.nu}")
        parts.append(f"center=({center_degrees[0]:.02f}, {center_degrees[1]:.02f})")
        parts.append(f"width={Angle(self.width).__repr__()}")

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

        for i, nu in enumerate(self.nu):
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
                nu=self.nu,
                center=self.center,
                frame=self.frame,
                degrees=False,
                units=units,
            )

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
        self.header["CTYPE3"] = "nu    "
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

    @property
    def x(self):
        return self.width * np.linspace(-0.5, 0.5, self.n_x)

    @property
    def y(self):
        return self.height * np.linspace(-0.5, 0.5, self.n_y)

    def plot(self, **kwargs):
        plot_map(self, **kwargs)
