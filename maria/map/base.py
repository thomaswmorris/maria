import arrow
import h5py
import os
import logging

import astropy as ap
import numpy as np
import scipy as sp

import dask.array as da
import time as ttime

from astropy.io import fits
from typing import Iterable

from ..calibration import Calibration
from ..units import parse_units

logger = logging.getLogger("maria")

here, this_filename = os.path.split(__file__)


STOKES = ["I", "Q", "U", "V"]


class Map:
    """
    The base class for maps. Maps have data
    """

    def __init__(
        self,
        data: float,
        weight: float | None,
        stokes: Iterable[str],
        nu: list[float],
        t: list[float],
        units: str = "K_RJ",
    ):

        self.data = da.asarray(data)
        self._weight = da.asarray(weight) if weight is not None else weight

        self.stokes = (
            [param.upper() for param in stokes] if stokes is not None else ["I"]
        )
        self.nu = np.atleast_1d(nu) if nu is not None else np.array([150.0])
        self.t = np.atleast_1d(t) if t is not None else np.array([ttime.time()])

        self.units = units

        parse_units(self.units)

        if len(self.stokes) != self.data.shape[0]:
            raise ValueError(
                f"'stokes' axis has length {len(self.stokes)} but map has shape (stokes, nu, t, y, x) = {self.data.shape}.",
            )

        if len(self.nu) != self.data.shape[1]:
            raise ValueError(
                f"'nu' axis has length {len(self.nu)} but map has shape (stokes, nu, t, y, x) = {self.data.shape}.",
            )

        if len(self.t) != self.data.shape[2]:
            raise ValueError(
                f"'time' axis has length {len(self.t)} but map has shape (stokes, nu, t, y, x) = {self.data.shape}.",
            )

    @property
    def nu_bins(self):
        """
        This might break in the year 3000.
        """
        return np.array([0, *(self.nu[1:] + self.nu[:-1]), 1e6])

    @property
    def nu_side(self):
        return (self.nu_bins[:-1] + self.nu_bins[1:]) / 2

    @property
    def t_bins(self):
        """
        This might break in the year 3000.
        """
        t_min = arrow.get("0001-01-01").timestamp()
        t_max = arrow.get("3000-01-01").timestamp()

        return np.array([t_min, *(self.t[1:] + self.t[:-1]), t_max])

    @property
    def t_side(self):
        return (self.t_bins[:-1] + self.t_bins[1:]) / 2

    @property
    def units_config(self):
        return parse_units(self.units)

    @property
    def weight(self):
        return (
            self._weight if self._weight is not None else da.ones(shape=self.data.shape)
        )

    @property
    def n_stokes(self):
        return self.data.shape[0]

    @property
    def n_nu(self):
        return self.data.shape[1]

    @property
    def n_t(self):
        return self.data.shape[2]

    @property
    def pixel_area(self):
        if hasattr(self, "resolution"):
            return self.resolution**2
        else:
            return self.x_res * self.y_res

    def to(self, units, inplace=False):

        if units == self.units:
            data = self.data.copy()

        else:
            data = np.zeros(self.data.shape)

            for i, nu in enumerate(self.nu):

                cal = Calibration(
                    f"{self.units} -> {units}",
                    nu=nu,
                    pixel_area=self.pixel_area,
                )
                data[i] = cal(self.data[i])

                if np.isnan(self.nu):
                    raise ValueError(f"Cannot convert map with frequency nu={nu}.")

        if inplace:
            self.data = data
            self.units = units

        else:
            package = self.package.copy()
            package.update({"data": data, "units": units})
            return type(self)(**package)

    def sample_nu(self, nu):

        map_nu_interpolator = sp.interpolate.interp1d(
            self.nu, self.data, axis=1, kind="linear"
        )

        nu_maps = []
        for nu in np.atleast_1d(nu):
            if nu < self.nu[0]:
                nu_maps.append(self.data[:, 0])
            elif not (nu < self.nu[-1]):  # this will include nan
                nu_maps.append(self.data[:, -1])
            else:
                nu_maps.append(map_nu_interpolator(nu))

        return np.stack(nu_maps, axis=1)

    @property
    def nu_bin_bounds(self):
        nu_boundaries = [0, *(self.nu[:-1] + self.nu[1:]) / 2, np.inf]
        return [(nu1, nu2) for nu1, nu2 in zip(nu_boundaries[:-1], nu_boundaries[1:])]

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

    def to_hdf(self, filename):

        with h5py.File(filename, "w") as f:

            f.create_dataset("data", dtype=float, data=self.data)

            if self._weight is not None:
                f.create_dataset("weight", dtype=float, data=self._weight)

            for field in ["nu", "t", "resolution", "center", "frame", "units"]:
                f.create_dataset(field, data=getattr(self, field))
