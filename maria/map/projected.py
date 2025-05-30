import copy
import logging
import os
from typing import Callable

import astropy as ap
import dask.array as da
import h5py
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from skimage.measure import block_reduce

from ..coords import Coordinates, frames
from ..units import Quantity, parse_units
from ..utils import compute_pointing_matrix, repr_phi_theta, unpack_implicit_slice
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
        z: float = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        cdelt1: float = None,
        cdelt2: float = None,
        center: tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
        dtype: type = np.float32,
    ):
        
        cdelt1 = cdelt1 or cdelt1
        cdelt2 = cdelt2 or cdelt2

        if data.ndim < 2:
            raise ValueError("'data' must be at least two-dimensional")

        if weight is not None:
            if weight.shape != data.shape:
                raise ValueError(f"'data' {data.shape} and 'weight' {weight.shape} should have the same shape.")
        else:
            weight = da.ones_like(data)

        map_dims = {"y": data.shape[-2], "x": data.shape[-1]}

        super().__init__(data=data, weight=weight, stokes=stokes, nu=nu, t=t, map_dims=map_dims, units=units, dtype=dtype)

        self.frame = frame

        if all(x is None for x in [width, height, resolution, cdelt1, cdelt2]):
            raise ValueError("You must pass at least one of 'width', 'height', 'resolution', 'cdelt1', or 'cdelt2'.")

        if cdelt1 is not None:
            cdelt2 = cdelt2 or cdelt1
        elif cdelt2 is not None:
            cdelt1 = cdelt1 or cdelt2
        elif resolution is not None:
            if not resolution > 0:
                raise ValueError("'resolution' must be positive.")
            cdelt1 = cdelt2 = resolution
        else:
            if width is not None:
                if not width > 0:
                    raise ValueError("'width' must be positive.")
                cdelt1 = width / self.n_x
                if height is not None:
                    cdelt2 = height / self.n_y
                else:
                    cdelt2 = cdelt1
            else:
                # here height must not be None
                cdelt1 = cdelt2 = height / self.n_y

        cdelt1 = cdelt1 if degrees else np.radians(cdelt1)
        cdelt2 = cdelt2 if degrees else np.radians(cdelt2)

        self.cdict = {
            "CTYPE": ["RA", "SIN"],
            "CUNIT": 2 * ["deg" if degrees else "rad"],
            "CDELT": [cdelt1, cdelt2],
            "CRVAL": Quantity(tuple(center), ("deg" if degrees else "rad")).deg,
            "CRPIX": [self.dims["x"] // 2, self.dims["y"] // 2],
            "NAXIS": [self.dims["x"], self.dims["y"]]
        }

        self.c = pd.DataFrame(self.cdict, index=[1, 2])

        # if self.cdelt2 < 0:
        #     self.data, self.weight = self.data[::-1], self.weight[::-1]
        #     self.cdelt2 *= -1
        # if self.cdelt1 < 0:
        #     self.data, self.weight = self.data[:, ::-1], self.weight[:, ::-1]
        #     self.cdelt1 *= -1

    def pointing_matrix(self, coords: Coordinates):
        offsets = coords.offsets(center=self.center, frame=self.frame)
        if "t" in self.dims:
            return compute_pointing_matrix(
                points=(offsets[..., 0], offsets[..., 1], coords._t), bins=(self.x_bins, self.y_bins[::-1], self.t_bins)
            )
        else:
            return compute_pointing_matrix(points=(offsets[..., 0], offsets[..., 1]), bins=(self.x_bins, self.y_bins[::-1]))

    @property
    def header(self):
        header = ap.io.fits.header.Header()

        header["SIMPLE"] = "T / conforms to FITS standard"
        header["BITPIX"] = "-32 / array data type"
        header["NAXIS"] = len(self.dims)
        for dim_index, n in enumerate(list(self.dims.values())):
            header[f"NAXIS{dim_index + 1}"] = n

        if "nu" in self.dims:
            header["RESTFREQ"] = self.nu.mean()

        header["CDELT1"] = np.degrees(self.cdelt1)  # degrees
        header["CDELT2"] = np.degrees(self.cdelt2)  # degrees

        header["CRPIX1"] = self.data.shape[-1] // 2
        header["CRPIX2"] = self.data.shape[-2] // 2
        header["BUNITS"] = self.units

        # specify x center
        header["CTYPE1"] = f"{self.frame_data["phi"].upper()}---SIN"
        header["CRVAL1"] = np.degrees(self.center[0])
        header["CUNIT1"] = "deg"

        # center y center
        header["CTYPE2"] = f"{self.frame_data["theta"].upper()}--SIN"
        header["CRVAL2"] = np.degrees(self.center[1])
        header["CUNIT2"] = "deg"

        return header

    def __getattr__(self, attr):
        broadcasted_attrs = ["STOKES", "NU", "T", "Y", "X"]
        if attr in broadcasted_attrs:
            broadcasted_attr_values = np.meshgrid(self.stokes, self.nu, self.t, self.y_side, self.x_side)
            return broadcasted_attr_values[broadcasted_attrs.index(attr)]

        raise AttributeError(f"'ProjectedMap' object has no attribute '{attr}'")

    def __getitem__(self, key):

        if not hasattr(key, "__iter__"):
            key = (key, )

        explicit_slices = unpack_implicit_slice(key, ndims=self.ndim)

        package = self.to("K_RJ").package()
        package["data"] = package["data"][key]
        package["weight"] = package["weight"][key]

        for axis, (dim, naxis) in enumerate(self.dims.items()):
            if dim in "xy":
                continue
            package[dim] = package[dim][explicit_slices[axis]]

        package["cdelt2"] *= explicit_slices[-2].step or 1
        package["cdelt1"] *= explicit_slices[-1].step or 1

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
            package["cdelt1"] *= self.n_x / new_n_x
            package["cdelt2"] *= self.n_y / new_n_y

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
  shape{self.dims_string}: {self.data.shape}
  stokes: {self.stokes if "stokes" in self.dims else "naive"}
  nu: {Quantity(self.nu, "Hz") if "nu" in self.dims else "naive"}
  t: {Quantity(self.t, "s") if "t" in self.dims else "naive"}
  quantity: {self.u["quantity"]}
  units: {self.units}
    min: {np.nanmin(self.data).compute():.03e}
    max: {np.nanmax(self.data).compute():.03e}
  center:
    {cphi_repr}
    {ctheta_repr}
  size(y, x): ({Quantity(self.height, "rad")}, {Quantity(self.width, "rad")})
  resolution(y, x): ({Quantity(self.cdelt2, "rad")}, {Quantity(self.cdelt1, "rad")})
  memory: {Quantity(self.data.nbytes + self.weight.nbytes, "B")}"""

    def package(self):
        package = copy.deepcopy(
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
                "cdelt1": np.degrees(self.cdelt1),
                "cdelt2": np.degrees(self.cdelt2),
            }
        )

        for dim in ["stokes", "nu", "t"]:
            if dim not in self.dims:
                package.pop(dim)

        return package

    @property
    def resolution(self):
        if not np.isclose(self.cdelt1, self.cdelt2, rtol=1e-3):
            RuntimeError(
                "Cannot return attribute 'resolution'; ProjectedMap has x-resolution"
                f" {np.degrees(self.cdelt1):.02f}° and y-resolution {np.degrees(self.cdelt1):.02f}°."
            )
        return self.cdelt1
    


    @property
    def center(self):
        return np.radians(self.cdict["CRVAL"])

    @property
    def cdelt1(self):
        return np.radians(self.cdict["CDELT"][0])

    @property
    def cdelt2(self):
        return np.radians(self.cdict["CDELT"][1])
    
    @property
    def width(self):
        return self.cdelt1 * self.n_x

    @property
    def height(self):
        return self.cdelt2 * self.n_y

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
        """
        The negative is so that
        """
        return -self.height * np.linspace(-0.5, 0.5, self.n_y + 1)

    @property
    def x_side(self):
        return (self.x_bins[:-1] + self.x_bins[1:]) / 2

    @property
    def y_side(self):
        return (self.y_bins[:-1] + self.y_bins[1:]) / 2
    
    @property
    def x(self):
        return self.x_side

    @property
    def y(self):
        return self.y_side

    def smooth(self, sigma: float = None, fwhm: float = None):
        if not (sigma is None) ^ (fwhm is None):
            raise ValueError("You must supply exactly one of 'sigma' or 'fwhm'.")

        package = self.package()

        sigma = sigma if sigma is not None else fwhm / np.sqrt(8 * np.log(2))
        x_sigma_pixels = sigma / self.cdelt1
        y_sigma_pixels = sigma / self.cdelt2
        package["data"] = sp.ndimage.gaussian_filter(self.data, sigma=(y_sigma_pixels, x_sigma_pixels), axes=(-2, -1))

        return type(self)(**package)

    def plot_slice(
        self,
        fig,
        ax,
        nu_index: int = 0,
        t_index: int = 0,
        stokes: str = "I",
        cmap: str = "default",
        rel_vmin: float = 0.005,
        rel_vmax: float = 0.995,
    ):
        d = self.data
        w = self.weight

        logger.debug(f"Plotting map slice (stokes={stokes}, nu_index={nu_index}, t_index={t_index})")

        if "stokes" in self.dims:
            if stokes not in self.stokes:
                raise ValueError(
                    f"Invalid stokes parameter '{stokes}'; available Stokes parameters for this map are {self.stokes}."
                )
            stokes_index = self.stokes.index(stokes)
            d, w = d[stokes_index], w[stokes_index]

        if "nu" in self.dims:
            d, w = d[nu_index], w[nu_index]

        if "t" in self.dims:
            d, w = d[t_index], w[t_index]

        if cmap == "default":
            cmap = "CMRmap" if stokes == "I" else "cmb"

        map_qdata = Quantity(d.compute(), units=self.units)
        subset = np.random.choice(d.size, size=min(d.size, 20000), replace=False)
        vmin, vmax = np.nanquantile(
            map_qdata.value.ravel()[subset],
            weights=w.ravel()[subset].compute(),
            q=[rel_vmin, rel_vmax],
            method="inverted_cdf",
        )

        # print(vmin, vmax)

        # assert False

        X = np.r_[self.x_bins, self.y_bins]
        grid_u = Quantity(X, "rad").u
        grid_center = Quantity(self.center, "rad")

        x = Quantity(self.x_bins, "rad")
        y = Quantity(self.y_bins, "rad")

        # fig = plt.figure(figsize=(6, 6), dpi=256, constrained_layout=True)
        # ax = fig.add_subplot(nx, ny, index, projection=WCS(header))

        ax.pcolormesh(
            getattr(x, grid_u["units"]),
            getattr(y, grid_u["units"]),
            map_qdata.value,
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
            pad=0,
        )

        label_parts = []
        if "stokes" in self.dims:
            label_parts.append(f"Stokes {stokes}")
        if "nu" in self.dims:
            label_parts.append(f"{Quantity(self.nu[nu_index], 'Hz')}")
        label_parts.append(rf"(${map_qdata.u['math_name']}$)")
        cbar.set_label(" ".join(label_parts), fontsize=10)
        ax.tick_params(axis="x", bottom=True, top=False)
        ax.tick_params(axis="y", left=True, right=False, rotation=90)
        ax.set_xlabel(rf"{self.frame_data['phi_long_name']}")
        ax.set_ylabel(rf"{self.frame_data['theta_long_name']}")

        ax.set_aspect("equal")

        return ax

    def plot(
        self,
        stokes: str = "I",
        nu_index: int = 0,
        t_index: int = 0,
        cmap: str = "default",
        rel_vmin: float = 0.005,
        rel_vmax: float = 0.995,
        filename: str = None,
    ):
        X = np.r_[self.x_bins, self.y_bins]
        grid_u = Quantity(X, "rad").u
        grid_center = Quantity(self.center, "rad")

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

        # nu_index = 0
        # t_index = 0
        # stokes = [["I", "Q"],
        #         ["U", "V"]]
        # stokes = ["I", "Q", "U"]
        # stokes = [["I", "Q"]]
        NU_INDEX, T_INDEX, STOKES = [np.atleast_2d(x) for x in np.broadcast_arrays(nu_index, t_index, stokes)]

        nrows, ncols = STOKES.shape

        max_fig_size = 12

        ax_size = 5  # np.minimum(np.maximum(np.minimum(max_fig_size / nrows, max_fig_size / ncols), 5), 8)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ax_size * ncols, ax_size * nrows),
            subplot_kw={"projection": WCS(header)},
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_1d(axes).reshape(STOKES.shape)

        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r, c]

                if c > 0:
                    ax.coords[1].set_ticks_visible(False)
                    ax.coords[1].set_ticklabel_visible(False)
                # if c == ncols - 1:
                #     ay2 = ax.secondary_yaxis("right")
                #     ay2.set_ylabel(rf"$\Delta\,\theta_y$ [{grid_u['units']}]")
                if r == 0:
                    ax2 = ax.secondary_xaxis("top")
                    ax2.set_xlabel(rf"$\Delta\,\theta$ [{grid_u['units']}]")
                if r < nrows - 1:
                    ax.coords[0].set_ticks_visible(False)
                    ax.coords[0].set_ticklabel_visible(False)

                if STOKES[r, c] is None:
                    continue

                self.plot_slice(
                    fig,
                    ax,
                    stokes=STOKES[r, c],
                    nu_index=NU_INDEX[r, c],
                    t_index=T_INDEX[r, c],
                    rel_vmin=rel_vmin,
                    rel_vmax=rel_vmax,
                    cmap=cmap,
                )

                if filename is not None:
                    plt.savefig(fname=filename, dpi=256)

    def to_fits(self, filepath):
        if self.dims.get("nu", np.nan) > 1:
            raise RuntimeError("Writing multifrequency to FITS is not supported")
        
        m = self.to(self.u["base_unit"])
        header = m.header
        # header["CDELT1"] *= -1
        header["CDELT2"] *= -1

        fits.writeto(
            filename=filepath,
            data=m.data[..., ::-1], # flip the y-axis to follow FITS convention
            header=header,
            overwrite=True,
        )

    def to_hdf(self, filename: str, compress: bool = True):
        compression_kwargs = {"compression": "gzip", "compression_opts": 9} if compress else {}

        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=np.float32, data=self.data, **compression_kwargs)
            if not (self.weight == 1).all().compute():
                f.create_dataset("weight", dtype=np.float32, data=self.weight, **compression_kwargs)
            for dim in ["stokes", "nu", "t"]:
                if dim in self.dims:
                    f.create_dataset(dim, data=getattr(self, dim))
            f.create_dataset("center", dtype=float, data=np.degrees(self.center))
            f.create_dataset("cdelt1", dtype=float, data=np.degrees(self.cdelt1))
            f.create_dataset("cdelt2", dtype=float, data=np.degrees(self.cdelt2))
            f.create_dataset("units", data=self.units)
            f.create_dataset("frame", data=self.frame)
