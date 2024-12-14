import os
import h5py

import astropy as ap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits

from ..coords import frames
from ..units import Angle


from .base import Map
from ..units import prefixes, QUANTITIES, parse_units


here, this_filename = os.path.split(__file__)


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
        resolution: float = None,
        center: tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
    ):

        # give it five dimensions
        data = data * np.ones((1, 1, 1, 1, 1))

        super().__init__(
            data=data, weight=weight, stokes=stokes, nu=nu, t=t, units=units
        )

        self.center = tuple(np.radians(center)) if degrees else center

        self.frame = frame

        parse_units(units)

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

        if len(self.nu) != self.n_nu:
            raise ValueError(
                f"Number of supplied nuuencies ({len(self.nu)}) does not match the "
                f"nu dimension of the supplied map ({self.n_nu}).",
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
        frame = frames[self.frame]
        center_degrees = np.degrees(self.center)

        parts.append(
            f"shape[stokes, nu, t, y, x]=({self.n_stokes}, {self.n_nu}, {self.n_t}, {self.n_y}, {self.n_x})",
        )
        parts.append(
            f"center[{frame['phi']}, {frame['theta']}]=({center_degrees[0]:.02f}°, {center_degrees[1]:.02f}°)",
        )
        parts.append(f"width={Angle(self.width).__repr__()}")

        return f"ProjectedMap({', '.join(parts)})"

    @property
    def weight(self):
        return (
            self._weight if self._weight is not None else np.ones(shape=self.data.shape)
        )

    @property
    def width(self):
        return self.resolution * self.n_x

    @property
    def height(self):
        return self.resolution * self.n_y

    @property
    def n_y(self):
        return self.data.shape[-2]

    @property
    def n_x(self):
        return self.data.shape[-1]

    @property
    def x_side(self):
        return self.width * np.linspace(-0.5, 0.5, self.n_x)

    @property
    def y_side(self):
        return self.height * np.linspace(-0.5, 0.5, self.n_y)

    @property
    def X(self):
        return np.meshgrid(self.x_side, self.y_side)[0]

    @property
    def Y(self):
        return np.meshgrid(self.x_side, self.y_side)[1]

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

        return ProjectedMap(
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

    def plot(
        self,
        filepath=None,
        nu_index=None,
        t_index=None,
        stokes="I",
        cmap="cmb",
        rel_vmin=0.001,
        rel_vmax=0.999,
        subplot_size=3,
    ):

        stokes_index = self.stokes.index(stokes)

        nu_index = (
            np.atleast_1d(nu_index) if nu_index is not None else np.arange(len(self.nu))
        )
        t_index = (
            np.atleast_1d(t_index) if t_index is not None else np.arange(len(self.t))
        )

        n_nu = len(nu_index)
        n_t = len(t_index)

        plot_width = np.maximum(12, subplot_size * n_nu)
        plot_height = np.maximum(12, subplot_size * n_t)
        plot_size = np.min([plot_width, plot_height, 5])

        # if (n_nu > 1) and (n_time > 1):
        fig, axes = plt.subplots(
            n_t,
            n_nu,
            figsize=(n_nu * plot_size, n_t * plot_size),
            sharex=True,
            sharey=True,
            constrained_layout=True,
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

        d = self.data.ravel()
        w = self.weight.ravel()
        subset = np.random.choice(d.size, size=10000)
        vmin, vmax = np.nanquantile(
            d[subset],
            weights=w[subset],
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
                # wcs_input = WCS(header, naxis=2)  # noqa F401

                # ax = fig.add_subplot(len(time_index), len(nu_index), i_ax, sharex=True)  # , projection=wcs_input)

                # ax.set_title(f"{nu} GHz")

                x = Angle(self.x_side)
                y = Angle(self.y_side)

                ax.pcolormesh(
                    x.values,
                    y.values,
                    self.data[stokes_index, i_nu, i_t].T[::-1],
                    cmap=cmap,
                    # interpolation="none",
                    # extent=map_extent,
                    vmin=vmin,
                    vmax=vmax,
                )

                if i_t == n_t - 1:
                    ax.set_xlabel(rf"$\Delta\,\theta_x$ [{x.units_short}.]")
                if i_nu == 0:
                    ax.set_ylabel(rf"$\Delta\,\theta_y$ [{y.units_short}.]")

        dummy_map = mpl.cm.ScalarMappable(
            mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
        )

        cbar = fig.colorbar(
            dummy_map,
            ax=axes,
            shrink=0.75,
            aspect=16,
            location="bottom",
        )

        u = parse_units(self.units)
        quantity = QUANTITIES.loc[u["quantity"]]
        units = (
            prefixes.loc[u["prefix"], "symbol_latex"] if u["prefix"] else ""
        ) + quantity.base_unit_latex
        cbar.set_label(f"{quantity.long_name} $[{units}]$")

        if filepath is not None:
            plt.savefig(filepath=filepath, dpi=256)
