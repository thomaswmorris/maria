import copy
import logging
import os
from typing import Callable

import astropy as ap
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy.wcs import WCS
from skimage.measure import block_reduce

from ..coords import frames
from ..units import Quantity, parse_units
from ..utils import repr_phi_theta, unpack_implicit_slice
from .base import Map

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class ProjectedMap(Map):
    """
    A rectangular map projected on the sphere. It has shape (stokes, nu, t, y, x).
    """

    def __init__(
        self,
        data: float,
        weight: float = None,
        stokes: float = None,
        nu: float = None,
        t: float = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        x_res: float = None,
        y_res: float = None,
        center: tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
        dtype: type = np.float32,
    ):
        # give it five dimensions
        data = data * np.ones((1, 1, 1, 1, 1))

        super().__init__(data=data, weight=weight, stokes=stokes, nu=nu, t=t, units=units, dtype=dtype)

        self.center = Quantity(tuple(center), ("deg" if degrees else "rad")).rad

        self.frame = frame

        parse_units(units)

        self.units = units

        if all(x is None for x in [width, height, resolution, x_res, y_res]):
            raise ValueError("You must pass at least one of 'width', 'height', 'resolution', 'x_res', or 'y_res'.")

        if x_res is not None:
            y_res = y_res or x_res
        elif y_res is not None:
            x_res = x_res or y_res
        elif resolution is not None:
            if not resolution > 0:
                raise ValueError("'resolution' must be positive.")
            x_res = y_res = resolution
        else:
            if width is not None:
                if not width > 0:
                    raise ValueError("'width' must be positive.")
                x_res = width / self.n_x
                if height is not None:
                    y_res = height / self.n_y
                else:
                    y_res = x_res
            else:
                # here height must not be None
                x_res = y_res = height / self.n_y

        self.x_res = np.radians(abs(x_res)) if degrees else x_res
        self.y_res = np.radians(y_res) if degrees else y_res

        if self.x_res < 0:
            raise ValueError()
        if self.y_res < 0:
            raise ValueError()

        if len(self.nu) != self.n_nu:
            raise ValueError(
                f"Number of supplied frequencies ({len(self.nu)}) does not match the "
                f"nu dimension of the supplied map ({self.n_nu}).",
            )

        # if self.units == "Jy/pixel":
        # self.to("K_RJ", inplace=True)

    @property
    def header(self):
        header = ap.io.fits.header.Header()

        header["SIMPLE"] = "T / conforms to FITS standard"
        header["BITPIX"] = "-32 / array data type"
        header["NAXIS"] = 5
        header["NAXIS1"] = self.n_y
        header["NAXIS2"] = self.n_x
        header["NAXIS3"] = self.n_t
        header["NAXIS4"] = self.n_nu
        header["NAXIS5"] = self.n_stokes

        header["RESTFREQ"] = self.nu.mean()

        header["CDELT1"] = -np.degrees(self.x_res)  # degrees
        header["CDELT2"] = np.degrees(self.y_res)  # degrees

        header["CRPIX1"] = self.data.shape[-1] // 2
        header["CRPIX2"] = self.data.shape[-2] // 2
        header["FRAME"] = self.frame
        header["BUNITS"] = self.units

        # specify x center
        header["CTYPE1"] = self.frame_data["phi"].upper()
        header["CRVAL1"] = np.degrees(self.center[0])
        header["CUNIT1"] = "deg     "

        # center y center
        header["CTYPE2"] = self.frame_data["theta"].upper()
        header["CRVAL2"] = np.degrees(self.center[1])
        header["CUNIT2"] = "deg     "

        return header

    def __getattr__(self, attr):
        broadcasted_attrs = ["STOKES", "NU", "T", "Y", "X"]
        if attr in broadcasted_attrs:
            broadcasted_attr_values = np.meshgrid(self.stokes, self.nu, self.t, self.y_side, self.x_side)
            return broadcasted_attr_values[broadcasted_attrs.index(attr)]

        raise AttributeError(f"'ProjectedMap' object has no attribute '{attr}'")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            package = self.to("K_RJ").package()
            package["data"] = package["data"][key]
            package["weight"] = package["weight"][key]

            stokes_slice, nu_slice, t_slice, y_slice, x_slice = unpack_implicit_slice(key)

            package["stokes"] = package["stokes"][stokes_slice]
            package["nu"] = package["nu"][nu_slice]
            package["t"] = package["t"][t_slice]
            package["x_res"] *= x_slice.step or 1
            package["y_res"] *= y_slice.step or 1

            return ProjectedMap(**package).to(self.units)

    def downsample(self, reduce: tuple, func: Callable = np.mean):
        if reduce:
            if len(reduce) != 4:
                raise ValueError("'reduce' must be a four-tuple of ints (nu, t, y, x)")

            package = self.to("K_RJ").package()
            package["data"] = block_reduce(package["data"].compute(), block_size=(1, *reduce), func=func)
            package["weight"] = block_reduce(package["weight"].compute(), block_size=(1, *reduce), func=func)

            *_, new_n_y, new_n_x = package["data"].shape

            package["nu"] = block_reduce(package["nu"], block_size=reduce[0])
            package["t"] = block_reduce(package["t"], block_size=reduce[1])
            package["x_res"] *= self.n_x / new_n_x
            package["y_res"] *= self.n_y / new_n_y

            return ProjectedMap(**package).to(self.units)

    @property
    def frame_data(self):
        return frames[self.frame]

    @property
    def points(self):
        return np.stack(np.meshgrid(self.y_side, self.x_side, indexing="ij"), axis=-1)

    def __repr__(self):
        cphi_repr, ctheta_repr = repr_phi_theta(*self.center, frame=self.frame)
        return f"""{self.__class__.__name__}:
  shape[stokes, nu, t, y, x]: {self.data.shape}
  stokes: {self.stokes}
  nu: {Quantity(self.nu, "Hz")}
  t: {Quantity(self.t, "s")}
  quantity: {self.u["quantity"]}
  units: {self.units}
    min: {np.nanmin(self.data).compute():.03e}
    max: {np.nanmax(self.data).compute():.03e}
  center:
    {cphi_repr}
    {ctheta_repr}
  size[y, x]: ({Quantity(self.height, "rad")}, {Quantity(self.width, "rad")})
  resolution[y, x]: ({Quantity(self.y_res, "rad")}, {Quantity(self.x_res, "rad")})
  memory: {Quantity(self.data.nbytes + self.weight.nbytes, "B")}"""

    def package(self):
        return copy.deepcopy(
            {
                "degrees": True,
                "data": self.data,
                "weight": self.weight,
                "stokes": self.stokes,
                "nu": self.nu,
                "t": self.t,
                "center": np.degrees(self.center),
                "frame": self.frame,
                "units": self.units,
                "x_res": np.degrees(self.x_res),
                "y_res": np.degrees(self.y_res),
            }
        )

    @property
    def resolution(self):
        if not np.isclose(self.x_res, self.y_res, rtol=1e-3):
            RuntimeError(
                "Cannot return attribute 'resolution'; ProjectedMap has x-resolution"
                f" {np.degrees(self.x_res):.02f}° and y-resolution {np.degrees(self.x_res):.02f}°."
            )
        return self.x_res

    @property
    def width(self):
        return self.x_res * self.n_x

    @property
    def height(self):
        return self.y_res * self.n_y

    @property
    def n_y(self):
        return self.data.shape[-2]

    @property
    def n_x(self):
        return self.data.shape[-1]

    @property
    def x_bins(self):
        return self.width * np.linspace(-0.5, 0.5, self.n_x + 1)

    @property
    def y_bins(self):
        return self.height * np.linspace(-0.5, 0.5, self.n_y + 1)

    @property
    def x_side(self):
        return (self.x_bins[:-1] + self.x_bins[1:]) / 2

    @property
    def y_side(self):
        return (self.y_bins[:-1] + self.y_bins[1:]) / 2

    def smooth(self, sigma: float = None, fwhm: float = None, inplace: bool = False):
        if not (sigma is None) ^ (fwhm is None):
            raise ValueError("You must supply exactly one of 'sigma' or 'fwhm'.")

        sigma = sigma if sigma is not None else fwhm / np.sqrt(8 * np.log(2))
        x_sigma_pixels = sigma / self.x_res
        y_sigma_pixels = sigma / self.y_res
        data = sp.ndimage.gaussian_filter(self.data, sigma=(0, 0, 0, y_sigma_pixels, x_sigma_pixels))

        if inplace:
            self.data = data

        else:
            return type(self)(
                data=data,
                weight=self.weight,
                width=self.width,
                t=self.t,
                nu=self.nu,
                center=self.center,
                frame=self.frame,
                degrees=False,
                units=self.units,
            )

    def plot(
        self,
        nu_index: int = 0,
        t_index: int = 0,
        stokes: str = "I",
        cmap: str = "CMRmap",
        rel_vmin: float = 0.005,
        rel_vmax: float = 0.995,
        filepath: str = None,
    ):
        stokes_index = self.stokes.index(stokes)

        map_qdata = Quantity(self.data.compute(), units=self.units)

        d = map_qdata.value.ravel()
        w = self.weight.ravel()
        subset = np.random.choice(d.size, size=10000)
        vmin, vmax = np.nanquantile(
            d[subset],
            weights=w[subset].compute(),
            q=[rel_vmin, rel_vmax],
            method="inverted_cdf",
        )

        X = np.r_[self.x_bins, self.y_bins]
        grid_u = Quantity(X, "rad").u
        grid_center = Quantity(self.center, "rad")

        x = Quantity(self.x_bins, "rad")
        y = Quantity(self.y_bins, "rad")

        header = fits.header.Header()
        header["CDELT1"] = -grid_u["factor"]
        header["CDELT2"] = grid_u["factor"]
        header["CRPIX1"] = 1
        header["CRPIX2"] = 1
        header["CTYPE1"] = "RA---SIN"
        header["CUNIT1"] = "deg     "
        header["CTYPE2"] = "DEC--SIN"
        header["CUNIT2"] = "deg     "
        header["RADESYS"] = "FK5     "
        header["CRVAL1"] = grid_center.deg[0]
        header["CRVAL2"] = grid_center.deg[1]

        fig = plt.figure(figsize=(6, 6), dpi=256, constrained_layout=True)
        ax = fig.add_subplot(projection=WCS(header))

        ax.pcolormesh(
            getattr(x, grid_u["units"]),
            getattr(y, grid_u["units"]),
            map_qdata.value[stokes_index, nu_index, t_index],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.grid(color="white", ls="solid", lw=5e-1)

        dummy_map = mpl.cm.ScalarMappable(
            mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
        )

        cbar = fig.colorbar(
            dummy_map,
            ax=ax,
            shrink=0.75,
            aspect=16,
            location="right",
        )

        qnu = Quantity(self.nu[nu_index], "Hz")
        cbar.set_label(rf"{map_qdata.q['long_name']} at {qnu} [${map_qdata.u['math_name']}$]")
        ax.tick_params(axis="x", bottom=True, top=False)
        ax.tick_params(axis="y", left=True, right=False, rotation=90)
        ax2 = ax.secondary_xaxis("top")
        ay2 = ax.secondary_yaxis("right")
        ax.set_xlabel(rf"{self.frame_data['phi_long_name']}")
        ax.set_ylabel(rf"{self.frame_data['theta_long_name']}")
        ax2.set_xlabel(rf"$\Delta\,\theta_x$ [${x.u['math_name']}$]")
        ay2.set_ylabel(rf"$\Delta\,\theta_y$ [${y.u['math_name']}$]")
        ax.set_aspect("equal")

        if filepath is not None:
            plt.savefig(filepath=filepath, dpi=256)

    def plot_many(
        self,
        nu_index=None,
        t_index=None,
        stokes="I",
        cmap="CMRmap",
        rel_vmin=0.005,
        rel_vmax=0.995,
        subplot_size=3,
        filepath=None,
    ):
        stokes_index = self.stokes.index(stokes)

        nu_index = np.atleast_1d(nu_index) if nu_index is not None else np.arange(len(self.nu))
        t_index = np.atleast_1d(t_index) if t_index is not None else np.arange(len(self.t))

        n_nu = len(nu_index)
        n_t = len(t_index)

        plot_width = np.maximum(12, subplot_size * n_nu)
        plot_height = np.maximum(12, subplot_size * n_t)
        plot_size = np.min([plot_width, plot_height, 4])

        # if (n_nu > 1) and (n_time > 1):
        fig, axes = plt.subplots(
            n_t,
            n_nu,
            figsize=(n_nu * plot_size, n_t * plot_size),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            dpi=256,
        )

        axes = np.atleast_1d(axes).reshape(n_nu, n_t)
        #     flat = False

        # else:
        #     n_rows = int(np.sqrt(n_maps))
        #     n_cols = int(np.ceil(n_maps / n_rows))
        #     fig, axes = plt.subplots(
        #         n_rows,
        #         n_cols,
        #         figsize=(6, 6),
        #         sharex=True,
        #         sharey=True,
        #         constrained_layout=True,
        #     )

        #     axes = np.atleast_1d(axes).ravel()
        #     flat = True

        # axes_generator = iter(axes.ravel())

        map_qdata = Quantity(self.data.compute(), units=self.units)

        d = map_qdata.value.ravel()
        w = self.weight.ravel()
        subset = np.random.choice(d.size, size=10000)
        vmin, vmax = np.nanquantile(
            d[subset],
            weights=w[subset].compute(),
            q=[rel_vmin, rel_vmax],
            method="inverted_cdf",
        )

        for i_t in t_index:
            for i_nu in nu_index:
                # ax = next(axes_generator) if flat else axes[i_nu, i_t]
                ax = axes[i_nu, i_t]

                nu = self.nu[i_nu]

                header = fits.header.Header()

                header["RESTFRQ"] = nu if nu > 0 else 150

                center_degrees = np.degrees(self.center)

                header["CDELT1"] = np.degrees(self.x_res)  # degree
                header["CDELT2"] = np.degrees(self.y_res)  # degree

                header["CRPIX1"] = self.n_x / 2
                header["CRPIX2"] = self.n_y / 2

                header["CTYPE1"] = "RA---SIN"
                header["CUNIT1"] = "deg     "
                header["CTYPE2"] = "DEC--SIN"
                header["CUNIT2"] = "deg     "
                header["RADESYS"] = "FK5     "

                header["CRVAL1"] = center_degrees[0]
                header["CRVAL2"] = center_degrees[1]
                # wcs_input = WCS(header, naxis=2)  # noqa F401

                # ax = fig.add_subplot(len(time_index), len(nu_index), i_ax, sharex=True)  # , projection=wcs_input)

                # ax.set_title(f"{nu} GHz")

                x = Quantity(self.x_bins, "rad")
                y = Quantity(self.y_bins, "rad")

                ax.pcolormesh(
                    x.value,
                    y.value,
                    map_qdata.value[stokes_index, i_nu, i_t],
                    cmap=cmap,
                    # interpolation="none",
                    # extent=map_extent,
                    vmin=vmin,
                    vmax=vmax,
                )

                if i_t == n_t - 1:
                    ax.set_xlabel(rf"$\Delta\,\theta_x$ [${x.u['math_name']}$]")
                if i_nu == 0:
                    ax.set_ylabel(rf"$\Delta\,\theta_y$ [${y.u['math_name']}$]")

                ax.set_aspect("equal")

        dummy_map = mpl.cm.ScalarMappable(
            mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
        )

        cbar = fig.colorbar(
            dummy_map,
            ax=axes,
            shrink=0.75,
            aspect=16,
            location="right",
        )

        cbar.set_label(f"{map_qdata.q['long_name']} [${map_qdata.u['math_name']}$]")

        if filepath is not None:
            plt.savefig(filepath=filepath, dpi=256)

    def to_fits(self, filepath):
        if self.n_nu > 1:
            raise RuntimeError("Cannot write multifrequency maps to FITS")

        m = self.to(self.u["base_unit"])
        header = self.header
        header["UNITS"] = m.units

        fits.writeto(
            filename=filepath,
            data=m.data,
            header=header,
            overwrite=True,
        )

    def to_hdf(self, filename: str, compress: bool = True):
        compression_kwargs = {"compression": "gzip", "compression_opts": 9} if compress else {}

        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=np.float32, data=self.data, **compression_kwargs)
            if not (self.weight == 1).all().compute():
                f.create_dataset("weight", dtype=np.float32, data=self.weight, **compression_kwargs)
            f.create_dataset("nu", dtype=float, data=self.nu)
            f.create_dataset("t", dtype=float, data=self.t)
            f.create_dataset("center", dtype=float, data=np.degrees(self.center))
            f.create_dataset("x_res", dtype=float, data=np.degrees(self.x_res))
            f.create_dataset("y_res", dtype=float, data=np.degrees(self.y_res))
            f.create_dataset("units", data=self.units)
            f.create_dataset("frame", data=self.frame)
