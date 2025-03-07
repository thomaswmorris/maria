import logging
import os

import astropy as ap
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits

from ..coords import frames
from ..units import QUANTITIES, Angle, parse_units, prefixes
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
        center: tuple[float, float] = (0.0, 0.0),
        frame: str = "ra_dec",
        degrees: bool = True,
        units: str = "K_RJ",
    ):
        # give it five dimensions
        data = data * np.ones((1, 1, 1, 1, 1))

        super().__init__(data=data, weight=weight, stokes=stokes, nu=nu, t=t, units=units)

        self.center = tuple(np.radians(center)) if degrees else center

        self.frame = frame

        parse_units(units)

        self.units = units

        if not ((width is not None) or (height is not None)) ^ (resolution is not None):
            raise ValueError("You must pass exactly one of ('width' and or 'height') or 'resolution'.")

        if resolution is not None:
            if not resolution > 0:
                raise ValueError("'resolution' must be positive.")
            self.x_res = np.radians(resolution) if degrees else resolution
            self.y_res = np.radians(resolution) if degrees else resolution
        else:
            if width is not None:
                if not width > 0:
                    raise ValueError("'width' must be positive.")
                width_radians = np.radians(width) if degrees else width
                self.x_res = width_radians / self.n_x
                if height is not None:
                    height_radians = np.radians(height) if degrees else height
                    self.y_res = height_radians / self.n_y
                else:
                    self.y_res = self.x_res
            else:
                # here height must not be None
                height_radians = np.radians(height) if degrees else height
                self.x_res = self.y_res = height_radians / self.n_y

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

        header["CDELT1"] = np.degrees(self.x_res)  # degrees
        header["CDELT2"] = np.degrees(self.y_res)  # degrees

        header["CRPIX1"] = self.data.shape[-1] // 2
        header["CRPIX2"] = self.data.shape[-2] // 2

        header["WIDTH"] = np.degrees(self.width)
        header["HEIGHT"] = np.degrees(self.height)
        header["FRAME"] = self.frame
        header["UNITS"] = self.units

        # specify x center
        header["CTYPE1"] = frames[self.frame]["phi"].upper()
        header["CRVAL1"] = np.degrees(self.center[0])
        header["CUNIT1"] = "deg     "

        # center y center
        header["CTYPE2"] = frames[self.frame]["theta"].upper()
        header["CRVAL2"] = np.degrees(self.center[1])
        header["CUNIT2"] = "deg     "

        return header

    def __getattr__(self, attr):
        broadcasted_attrs = ["STOKES", "NU", "T", "Y", "X"]
        if attr in broadcasted_attrs:
            broadcasted_attr_values = np.meshgrid(self.stokes, self.nu, self.t, self.y_side, self.x_side)
            return broadcasted_attr_values[broadcasted_attrs.index(attr)]

        raise AttributeError(f"'ProjectedMap' object has no attribute '{attr}'")

    @property
    def points(self):
        return np.stack(np.meshgrid(self.y_side, self.x_side, indexing="ij"), axis=-1)

    # broadcasted_attrs = ["NU", "T", "Y", "X"]

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
        parts.append(f"height={Angle(self.height).__repr__()}")

        return f"ProjectedMap({', '.join(parts)})"

    @property
    def package(self):
        package_keys = [
            "data",
            "weight",
            "stokes",
            "nu",
            "t",
            "width",
            "height",
            "center",
            "frame",
            "units",
        ]
        return {"degrees": False, **{k: getattr(self, k) for k in package_keys}}

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
                height=self.height,
                t=self.t,
                nu=self.nu,
                center=self.center,
                frame=self.frame,
                degrees=False,
                units=self.units,
            )

    def downsample(self, n_x=None, n_y=None):
        """
        TODO: implement t and nu downsampling
        """

        data = self.to("K_RJ").data

        new_n_nu = self.n_nu
        new_n_t = self.n_t
        new_n_y = n_y or self.n_y
        new_n_x = n_x or self.n_x

        # new_nu_bins = np.linspace(self.nu_bins.min(), self.nu_bins.max(), new_n_nu + 1)
        # new_t_bins = np.linspace(self.t_bins.min(), self.t_bins.max(), new_n_t + 1)
        new_y_bins = np.linspace(self.y_bins.min(), self.y_bins.max(), new_n_y + 1)
        new_x_bins = np.linspace(self.x_bins.min(), self.x_bins.max(), new_n_x + 1)
        bins_tuple = (new_y_bins, new_x_bins)

        new_data = np.zeros((len(self.stokes), new_n_nu, new_n_t, new_n_y, new_n_x))

        for stokes_index, stokes in enumerate(self.stokes):
            for nu_index, nu in enumerate(self.nu):
                for t_index, t in enumerate(self.nu):
                    bs = sp.stats.binned_statistic_dd(
                        sample=self.points.reshape(-1, 2),
                        values=data[stokes_index, nu_index, t_index].reshape(-1),
                        bins=bins_tuple,
                        statistic="mean",
                    )

                    new_data[stokes_index, nu_index, t_index] = bs.statistic

        return ProjectedMap(
            data=new_data,
            t=self.t,
            nu=self.nu,
            width=self.width,
            height=self.height,
            center=self.center,
            frame=self.frame,
            degrees=False,
            units="K_RJ",
        ).to(units=self.units)

    def plot(
        self,
        nu_index=None,
        t_index=None,
        stokes="I",
        cmap="cmb",
        rel_vmin=0.001,
        rel_vmax=0.999,
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
            d[subset].compute(),
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

                x = Angle(self.x_bins)
                y = Angle(self.y_bins)

                ax.pcolormesh(
                    x.values,
                    y.values,
                    self.data[stokes_index, i_nu, i_t],
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
            location="bottom",
        )

        u = parse_units(self.units)
        quantity = QUANTITIES.loc[u["quantity"]]
        units = (prefixes.loc[u["prefix"], "symbol_latex"] if u["prefix"] else "") + quantity.base_unit_latex
        cbar.set_label(f"{quantity.long_name} $[{units}]$")

        if filepath is not None:
            plt.savefig(filepath=filepath, dpi=256)

    def to_fits(self, filepath):
        m = self.to(self.units_config["base_unit"])
        header = self.header
        header["UNITS"] = m.units

        fits.writeto(
            filename=filepath,
            data=m.data,
            header=header,
            overwrite=True,
        )

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=float, data=self.data)

            if self._weight is not None:
                f.create_dataset("weight", dtype=float, data=self._weight)

            for field in ["nu", "t", "width", "height", "center", "frame", "units"]:
                f.create_dataset(field, data=getattr(self, field))
