import os
import h5py

import astropy as ap
import numpy as np
import scipy as sp

import dask.array as da

from astropy.io import fits
from typing import Iterable

from ..instrument import Band
from ..constants import k_B
from ..units import Angle, Calibration, parse_units  # noqa

# from ..plotting import plot_map

# from astropy.wcs import WCS


here, this_filename = os.path.split(__file__)


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
        self.nu = np.atleast_1d(nu) if nu is not None else np.array([np.nan])
        self.t = np.atleast_1d(t) if t is not None else np.array([np.nan])

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

    def to(self, units, inplace=False):

        if units == self.units:
            data = self.data.copy()

        else:
            data = np.zeros(self.data.shape)

            for i, nu in enumerate(self.nu):

                cal = Calibration(
                    f"{self.units} -> {units}", nu=1e9 * nu, res=self.resolution
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

    def power(self, band: Band):

        if self.units_config["quantity"] in [
            "rayleigh_jeans_temperature",
            "spectral_flux_density_per_pixel",
        ]:

            nu_boundaries = [
                band.nu.min(),
                *(self.nu[1:] + self.nu[:-1]) / 2,
                band.nu.max(),
            ]
            nu_K_RJ_data = self.to("K_RJ").data[0]

            power_map_pW = 0
            for nu_index, (nu1, nu2) in enumerate(
                zip(nu_boundaries[:-1], nu_boundaries[1:])
            ):

                dummy_nu = np.linspace(nu1, nu2, 1024)
                tau_integral = band.efficiency * np.trapezoid(
                    band.passband(dummy_nu), x=1e9 * dummy_nu
                )
                power_map_pW += 1e12 * k_B * nu_K_RJ_data[nu_index] * tau_integral

            return power_map_pW

        else:
            raise ValueError()

        # for nu1, nu2 in zip(band.nu[:-1], band.nu[1:]):
        #     nu = (nu1 + nu2) / 2
        #     width_Hz = 1e9 * (nu2 - nu1)
        #     nu_map = self.sample_nu(nu)[0, 0] # this has shape (t, x, y)
        #     power_map += 1e12 * k_B * band.efficiency * band.passband(nu) * nu_map * width_Hz

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

    def downsample(self, shape):
        zoom_factor = np.array(shape) / self.data.shape

        return type(self)(
            data=sp.ndimage.zoom(self.data, zoom=zoom_factor),
            t=sp.ndimage.zoom(self.t, zoom=zoom_factor[0]),
            nu=sp.ndimage.zoom(self.nu, zoom=zoom_factor[1]),
            width=self.width,
            center=self.center,
            frame=self.frame,
            degrees=False,
        )

    def to_hdf(self, filename):

        with h5py.File(filename, "w") as f:

            f.create_dataset("data", dtype=float, data=self.data)

            if self._weight is not None:
                f.create_dataset("weight", dtype=float, data=self._weight)

            for field in ["nu", "t", "resolution", "center", "frame", "units"]:
                f.create_dataset(field, data=getattr(self, field))
