import copy
import logging
import os
from typing import Callable, Iterable

import astropy as ap
import dask.array as da
import h5py
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy.wcs import WCS
from skimage.measure import block_reduce

from ..array import Array
from ..coords import Coordinates, Frame
from ..io import FITS_DEFAULT_UNITS, FITS_TYPE_ALIASES, repr_phi_theta
from ..units import Quantity, parse_units
from ..utils import compute_pointing_matrix_ingredients, unpack_implicit_slice
from .base import Map

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")


class ProjectionMap(Map):
    """
    A rectangular map projected on the sphere. It has shape (stokes, nu, t, y, x).
    """

    def __init__(
        self,
        data: float,
        units: str,
        weight: float = None,
        stokes: float = None,
        nu: Iterable[float] = None,
        t: Iterable[float] = None,
        v: Iterable[float] = None,
        z: Iterable[float] = None,
        eta: Iterable[float] = None,
        xi: Iterable[float] = None,
        width: float = None,
        height: float = None,
        resolution: float = None,
        x_res: float = None,
        y_res: float = None,
        center: tuple[float, float] = (0.0, 0.0),
        beam: tuple[float, float, float] = 0.0,
        frame: str = "ra/dec",
        degrees: bool = True,
        dtype: type = np.float32,
    ):
        # give it five dimensions

        if data.ndim < 2:
            raise ValueError("'data' must be at least two-dimensional")

        if weight is not None:
            if weight.shape != data.shape:
                raise ValueError(f"'data' {data.shape} and 'weight' {weight.shape} should have the same shape.")
        else:
            weight = da.ones_like(data)

        map_dims = {"eta": data.shape[-2], "xi": data.shape[-1]}

        try:
            assert len(center) == 2
            self.center = tuple([*Quantity(center, ("deg" if degrees else "rad")).pin("deg")])
        except Exception:
            raise ValueError("'center' must be a two-tuple of numbers")

        super().__init__(
            data=data,
            weight=weight,
            stokes=stokes,
            nu=nu,
            t=t,
            z=z,
            v=v,
            beam=beam,
            map_dims=map_dims,
            units=units,
            frame=frame,
            degrees=degrees,
            dtype=dtype,
        )

        # from the inputs, construct xi and eta
        if (xi is None) or (eta is None):
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
                    x_res = width / data.shape[-1]
                    if height is not None:
                        y_res = height / data.shape[-2]
                    else:
                        y_res = x_res
                elif height is not None:
                    # here height must not be None
                    x_res = y_res = height / data.shape[-2]
                else:
                    raise ValueError(
                        f"{self.__class__.__name__} needs one of ['width', 'height', 'resolution', 'x_res', 'y_res']."
                    )

            xi = x_res * map_dims["xi"] * np.linspace(-0.5, 0.5, self.dims["xi"])
            eta = -y_res * map_dims["eta"] * np.linspace(-0.5, 0.5, self.dims["eta"])

        self.xi = Quantity(xi, "deg" if degrees else "rad")
        self.eta = Quantity(eta, "deg" if degrees else "rad")

        # apply parity convention
        self.apply_parity(**{dim: (-1 if dim in ["v", "eta"] else 1) for dim in self.dims if dim not in ["stokes"]})

    def apply_parity(self, **parity):

        parity_slicing = []
        for dim in parity:
            if dim not in self.dims:
                raise ValueError(f"'{dim}' not in map axes")
            dim_parity = parity[dim]
            dim_values = getattr(self, dim)
            if dim_values.size > 1:
                grad = np.gradient(dim_values.base_units_value)
                if all(grad < 0):
                    dim_parity *= -1
                elif not all(grad > 0):
                    raise ValueError(f"Values for supplied axis '{dim}' are not monotonic")
            setattr(self, dim, dim_values[::dim_parity])
            parity_slicing.append(slice(None, None, dim_parity))

        self.data = self.data[tuple(parity_slicing)]

        if self._weight is not None:
            self.weight = self.weight[tuple(parity_slicing)]

    def _pointing_matrix_ingredients(self, coords: Coordinates, bilinear: bool = True):
        offsets = coords.offsets(center=(self.center[0].rad, self.center[1].rad), frame=self.frame.name)

        return compute_pointing_matrix_ingredients(
            x_list=(
                coords._t,
                offsets[..., 1],
                offsets[..., 0],
            ),
            side_list=(self.t.seconds, self.eta.radians, self.xi.radians),
            bilinear=bilinear,
        )

    def _stokes_weighted_pointing_matrix_ingredients(self, coords: Coordinates, dets: Array, bilinear: bool = True):

        M = dets.mueller()
        samples, pixels, weights, n_pixels, n_samples = self._pointing_matrix_ingredients(coords=coords, bilinear=bilinear)

        if "nu" in self.dims:
            for nu_index, nu in enumerate(self.nu):
                pixels[:, dets.band_center == nu.Hz] += nu_index * n_pixels
            n_pixels *= self.dims["nu"]

        stokes_list = self.stokes if "stokes" in self.dims else "I"

        samples_list, pixels_list, weights_list = [], [], []
        for stokes_index, stokes in enumerate(stokes_list):
            samples_list.append(samples)
            pixels_list.append(pixels + n_pixels * stokes_index)
            weights_list.append(weights * M[:, 0, "IQUV".index(stokes)][:, None])

        return (
            np.concatenate(weights_list).ravel(),
            np.concatenate(samples_list).ravel(),
            np.concatenate(pixels_list).ravel(),
            n_samples,
            len(stokes_list) * n_pixels,
        )

    def stokes_weighted_pointing_matrix(self, coords: Coordinates, dets: Array, bilinear: bool = True):

        weights, samples, pixels, n_samples, n_pixels = self._stokes_weighted_pointing_matrix_ingredients(
            coords=coords, dets=dets, bilinear=bilinear
        )

        return sp.sparse.csr_array((weights, (samples, pixels)), shape=(n_samples, n_pixels))

    def header(self):

        averaged_beam = self.beam.mean(axis=tuple(range(len(self.dims) - 2)))

        header = ap.io.fits.header.Header()
        header["SIMPLE"] = "T / conforms to FITS standard"
        header["BITPIX"] = "-32 / array data type"
        header["NAXIS"] = len(self.dims)
        for dim_index, n in enumerate(list(self.dims.values())[::-1]):
            header[f"NAXIS{dim_index + 1}"] = n

        header["BUNIT"] = self.units
        header["BMAJOR"] = averaged_beam[0].degrees.item()
        header["BMINOR"] = averaged_beam[1].degrees.item()
        header["BPA"] = averaged_beam[2].degrees.item()

        CTYPE1 = self.frame.fits_phi
        header["CTYPE1"] = f"{CTYPE1}{(5 - len(CTYPE1)) * '-'}SIN"
        header["CRVAL1"] = self.center[0].deg
        header["CRPIX1"] = self.xi.size // 2
        header["CDELT1"] = -self.xi_res.deg  # longitude goes the other way
        header["CUNIT1"] = "deg"

        CTYPE2 = self.frame.fits_theta
        header["CTYPE2"] = f"{CTYPE2}{(5 - len(CTYPE2)) * '-'}SIN"
        header["CRVAL2"] = self.center[1].deg
        header["CRPIX2"] = self.eta.size // 2
        header["CDELT2"] = self.eta_res.deg
        header["CUNIT2"] = "deg"

        for dim_index, dim in enumerate(list(self.dims)[:-2][::-1]):
            if dim == "stokes":
                continue

            AXIS = dim_index + 3

            grad = getattr(self, dim)[0]

            units = FITS_DEFAULT_UNITS[dim]
            dim_values = getattr(self, dim).to(units)
            if dim_values.size > 1:
                grad = np.gradient(dim_values)
                delt = np.median(grad)
                if not grad.std() / np.abs(delt) < 1e-6:
                    raise RuntimeError("Cannot write irregular maps to FITS")
            else:
                delt = 0.0

            header[f"CTYPE{AXIS}"] = FITS_TYPE_ALIASES[dim][0]
            header[f"CRVAL{AXIS}"] = dim_values[0]
            header[f"CRPIX{AXIS}"] = 0
            header[f"CDELT{AXIS}"] = delt
            header[f"CUNIT{AXIS}"] = units

        return header

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)

        explicit_slices = unpack_implicit_slice(key, ndims=self.ndim)
        package = self.package()

        package["data"] = package["data"][key]
        if self._weight is not None:
            package["weight"] = package["weight"][key]
        package["beam"] = package["beam"][explicit_slices[: -len(self.map_dims)]]

        for axis, (dim, naxis) in enumerate(self.dims.items()):
            if isinstance(explicit_slices[axis], int):
                package.pop(dim)
            else:
                package[dim] = package[dim][explicit_slices[axis]]

        x_res_factor = explicit_slices[-1].step or 1.0
        y_res_factor = explicit_slices[-2].step or 1.0

        dimensions = parse_units(self.units)["dimension_vector"]

        # downsampling changes the pixel area, so we might have to adjust
        package["data"] *= (x_res_factor * y_res_factor) ** dimensions.pixel

        return ProjectionMap(**package)

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

            return ProjectionMap(**package).to(self.units)

    # @property
    # def points(self):
    #     return np.stack(np.meshgrid(self.y_side, self.x_side, indexing="ij"), axis=-1)

    def __repr__(self):
        # beam_repr = self.beam
        center_repr = "center:"
        for key, value in repr_phi_theta(self.center[0].rad, self.center[1].rad, frame=self.frame.name).items():
            center_repr += f"\n    {key}: {value}"
        return f"""{self.__class__.__name__}:
{self.__repr_base__()}
  eta({self.dims["eta"]}):
    height: {self.height}
    res: {self.eta_res}
  xi({self.dims["xi"]}): 
    width: {self.width}
    res: {self.xi_res}
  frame: {self.frame.name}
  {center_repr}
  beam(maj, min, psi): {self.beam_repr()}
  memory: {Quantity(self.data.nbytes + self.weight.nbytes, "B")}"""

    def package(self):
        package = copy.deepcopy(
            {
                "data": self.data,
                "weight": self.weight,
                "center": (self.center[0].deg, self.center[1].deg),
                "frame": self.frame.name,
                "units": self.units,
                "beam": self.beam.deg,
                "degrees": True,
            }
        )

        for dim in self.dims:
            package[dim] = getattr(self, dim)

        return package

    @property
    def pixel_area(self):
        return Quantity(abs(self.xi_res.rad * self.eta_res.rad), "sr")

    @property
    def resolution(self):
        if not np.isclose(self.xi_res.rad, self.eta_res.rad, rtol=1e-3):
            RuntimeError(
                "Cannot return attribute 'resolution'; ProjectionMap has x-resolution"
                f" {self.xi_res}° and y-resolution {self.eta_res}°."
            )
        return self.xi_res

    @property
    def width(self):
        return self.xi.max() - self.xi.min()

    @property
    def height(self):
        return self.eta.max() - self.eta.min()

    @property
    def n_xi(self):
        return len(self.xi)

    @property
    def xi_res(self):
        if not hasattr(self, "_xi_res"):
            xi_grad = np.gradient(self.xi.rad)
            med_xi_res = np.median(xi_grad)
            if np.std(xi_grad) / med_xi_res > 1e-4:
                self._xi_res = "irregular"
            else:
                self._xi_res = Quantity(med_xi_res, "rad")
        return self._xi_res

    @property
    def n_eta(self):
        return len(self.eta)

    @property
    def eta_res(self):
        if not hasattr(self, "_eta_res"):
            eta_grad = np.gradient(self.eta.rad)
            med_eta_res = np.median(eta_grad)
            if np.std(eta_grad) / med_eta_res > 1e-4:
                self._eta_res = "irregular"
            else:
                self._eta_res = Quantity(med_eta_res, "rad")
        return self._eta_res

    # @property
    # def xi_bins(self):
    #     """
    #     Following array indexing conventions,
    #     """
    #     self.x_res.rad
    #     return self.width.rad * np.linspace(-0.5, 0.5, self.n_x + 1)

    # @property
    # def y_bins(self):
    #     """
    #     The negative is so that follows indexing rules
    #     """
    #     return -self.height.rad * np.linspace(-0.5, 0.5, self.n_y + 1)

    @property
    def x_side(self):
        logger.warning("Attribute 'x_side' is deprecated, use 'xi' instead")
        return self.xi.rad

    @property
    def y_side(self):
        logger.warning("Attribute 'y_side' is deprecated, use 'eta' instead")
        return self.eta.rad

    def smooth(self, sigma: float = None, fwhm: float = None):
        if not (sigma is None) ^ (fwhm is None):
            raise ValueError("You must supply exactly one of 'sigma' or 'fwhm'.")

        package = self.package()

        sigma = sigma if sigma is not None else fwhm / np.sqrt(8 * np.log(2))
        x_sigma_pixels = sigma / self.xi_res
        y_sigma_pixels = sigma / self.eta_res

        numer = sp.ndimage.gaussian_filter(self.data * self.weight, sigma=(y_sigma_pixels, x_sigma_pixels), axes=(-2, -1))
        denom = sp.ndimage.gaussian_filter(self.weight, sigma=(y_sigma_pixels, x_sigma_pixels), axes=(-2, -1))

        package["data"] = numer / denom
        package["weight"] = numer

        return type(self)(**package)

    def transfer_function(
        self,
        input_map=None,
        n_bins: int = 20,
        stokes: str = "I",
        nu_index: int = 0,
        t_index: int = 0,
        window: bool = True,
    ):
        """Compute the spatial transfer function relative to an input map.

        Returns a :class:`TransferFunction` object whose ``.plot()`` method
        produces a three-panel figure (input map, output map, transfer function).

        When the map was produced by a mapper whose TODs came from a
        :class:`~maria.Simulation` with ``map=...``, the input map is
        propagated automatically and this argument can be omitted.

        Parameters
        ----------
        input_map : ProjectionMap, optional
            The input sky map injected into the simulation.  Falls back to the
            map stored on this object (``_input_map``) when not given.
        n_bins : int
            Number of logarithmically-spaced spatial frequency bins.
        stokes : str
            Stokes parameter to use ("I", "Q", "U", or "V").
        nu_index : int
            Frequency channel index for multi-channel maps.
        t_index : int
            Time index for time-varying maps.
        window : bool
            Apply a 2D Hann window before FFT to reduce spectral leakage.

        Returns
        -------
        TransferFunction
        """
        from .transfer import TransferFunction, compute_transfer_function

        if input_map is None:
            input_map = getattr(self, "_input_map", None)
        if input_map is None:
            raise ValueError(
                "No input map available. Pass input_map explicitly or run the "
                "simulation with map=<ProjectionMap>."
            )

        u, T = compute_transfer_function(
            input_map,
            self,
            n_bins=n_bins,
            stokes=stokes,
            nu_index=nu_index,
            t_index=t_index,
            window=window,
        )
        return TransferFunction(u=u, T=T, input_map=input_map, output_map=self)

    def plot_slice(
        self,
        fig,
        ax,
        nu_index: int = 0,
        t_index: int = 0,
        v_index: int = 0,
        stokes: str = "I",
        cmap: str = "default",
        norm_method: str = "slice",
        contrast: float = 1e-3,
        rel_vmin: float = None,
        rel_vmax: float = None,
        vmin: float = None,
        vmax: float = None,
        center_zero: bool = False,
    ):
        d = self.data
        w = self.weight

        logger.debug(f"Plotting map slice (stokes={stokes}, nu_index={nu_index}, t_index={t_index})")

        if "stokes" in self.dims:
            if stokes not in self.stokes:
                raise ValueError(f"Invalid stokes parameter '{stokes}'. This map has stokes parameters '{self.stokes}'.")
            stokes_index = self.stokes.index(stokes)
            d, w = d[stokes_index], w[stokes_index]

        if "nu" in self.dims:
            try:
                d, w = d[nu_index], w[nu_index]
            except IndexError:
                raise IndexError(
                    f"nu_index={nu_index} is out of bounds; map has shape {self.dims_string} = {tuple(self.dims.values())}"
                )

        if "v" in self.dims:
            try:
                d, w = d[v_index], w[v_index]
            except IndexError:
                raise IndexError(
                    f"v_index={v_index} is out of bounds; map has shape {self.dims_string} = {tuple(self.dims.values())}"
                )

        if "t" in self.dims:
            try:
                d, w = d[t_index], w[t_index]
            except IndexError:
                raise IndexError(
                    f"t_index={t_index} is out of bounds; map has shape {self.dims_string} = {tuple(self.dims.values())}"
                )

        if cmap == "default":
            cmap = "CMRmap" if stokes == "I" else "cmb"

        map_slice_qdata = Quantity(d.compute(), units=self.units)
        map_slice_values = map_slice_qdata.human_value

        rel_vmin = rel_vmin or contrast
        rel_vmax = rel_vmax or 1.0 - contrast

        if vmin is None and vmax is None:
            if norm_method == "slice":
                norm_values = map_slice_values
                norm_weights = w.compute()
            else:
                raise ValueError(f"Invalid norm_method '{norm_method}'")

            subset = np.random.choice(norm_values.size, size=min(norm_values.size, 100000), replace=False)
            vmin, vmax = np.nanquantile(
                norm_values.ravel()[subset],
                weights=norm_weights.ravel()[subset],
                q=(rel_vmin, rel_vmax),
                method="inverted_cdf",
            )
            if center_zero:
                abs_max = max(vmax, -vmin)
                vmin, vmax = -abs_max, abs_max

            logger.debug(f"Inferring (vmin, vmax) = ({vmin}, {vmax})")

        X = np.r_[self.x_bins, self.y_bins]
        qX = Quantity(X, "rad")
        grid_u, grid_hu = qX.u, qX.hu

        x = Quantity(self.x_bins, "rad")
        y = Quantity(self.y_bins, "rad")

        # fig = plt.figure(figsize=(6, 6), dpi=256, constrained_layout=True)
        # ax = fig.add_subplot(nx, ny, index, projection=WCS(header))

        ax.pcolormesh(
            getattr(x, grid_hu["units"]),
            getattr(y, grid_hu["units"]),
            map_slice_qdata.human_value,
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

        label = rf"${map_slice_qdata.hu['math_name']}$"
        if "stokes" in self.dims:
            label += f" Stokes {stokes}"
        if "nu" in self.dims:
            label += f" at {self.nu[nu_index]}"
        cbar.set_label(label, fontsize=10)
        ax.tick_params(axis="x", bottom=True, top=False)
        ax.tick_params(axis="y", left=True, right=False, rotation=90)

        lon = ax.coords[0]
        lon.set_axislabel(rf"{self.frame.phi_long_name}")
        lon.set_ticks_position("b")
        lon.set_ticklabel_position("b")
        lon.set_axislabel_position("b")

        lat = ax.coords[1]
        lat.set_axislabel(rf"{self.frame.theta_long_name}")
        lat.set_ticks_position("l")
        lat.set_ticklabel_position("l")
        lat.set_axislabel_position("l")

        ax.set_aspect("equal")

        return ax

    def plot(
        self,
        slices: dict = {},
        cmap: str = "cmb",
        units: str = None,
        filename: str = None,
        contrast: str = 1e-3,
        center_zero: bool = False,
        vmin: float = None,
        vmax: float = None,
        rel_vmin: float = None,
        rel_vmax: float = None,
        ax_size: float = 5.0,
    ):

        rel_vmin = rel_vmin or contrast
        rel_vmax = rel_vmax or 1.0 - contrast

        dim_slices = {dim: slices.get(dim, [0]) for dim in self.dims}
        dim_slices["eta"] = slice(None, None, None)
        dim_slices["xi"] = slice(None, None, None)

        SLICES = {dim: np.atleast_2d(x) for dim, x in zip(self.dims, np.broadcast_arrays(*dim_slices.values()))}

        nrows, ncols = SLICES["xi"].shape

        if units is None:
            units = Quantity(self.data, self.units).human_units
            logger.debug(f"Plotting with units '{units}'")

        map_data = self.to(units).data.compute()
        u = parse_units(units)

        grid_hu = Quantity(np.r_[self.xi.rad, self.eta.rad], "rad").hu

        lon_frame = f"{Frame(self.frame).fits['phi']}"
        lat_frame = f"{Frame(self.frame).fits['theta']}"

        header = fits.header.Header()
        header["CDELT1"] = -np.degrees(grid_hu["base_units_factor"])
        header["CDELT2"] = np.degrees(grid_hu["base_units_factor"])
        header["CRPIX1"] = 1
        header["CRPIX2"] = 1
        header["CTYPE1"] = f"{lon_frame}{(5 - len(lon_frame)) * '-'}SIN"
        header["CUNIT1"] = "deg     "
        header["CTYPE2"] = f"{lat_frame}{(5 - len(lat_frame)) * '-'}SIN"
        header["CUNIT2"] = "deg     "
        header["CRVAL1"] = self.center[0].deg
        header["CRVAL2"] = self.center[1].deg

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ax_size * ncols * 1.2, ax_size * nrows),
            subplot_kw={"projection": WCS(header)},
            constrained_layout=True,
            sharex=False,
            sharey=False,
            # ppi=256,
        )
        axes = np.atleast_2d(axes)

        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]
                ax_slices = {dim: SLICES[dim][i, j] for dim in self.dims}

                if "stokes" in ax_slices:
                    stokes = ax_slices["stokes"]
                    if isinstance(stokes, str):
                        if stokes not in self.stokes:
                            raise ValueError(f"Map does not have stokes parameter '{stokes}'")
                        ax_slices["stokes"] = self.stokes.index(stokes)

                map_slice_data = map_data[tuple(ax_slices.values())]
                map_slice_weights = self.weight[tuple(ax_slices.values())].compute()

                if vmin is None or vmax is None:
                    subset = np.random.choice(map_slice_data.size, size=min(map_slice_data.size, 100000), replace=False)
                    slice_vmin, slice_vmax = np.nanquantile(
                        map_slice_data.ravel()[subset],
                        weights=map_slice_weights.ravel()[subset],
                        q=(rel_vmin, rel_vmax),
                        method="inverted_cdf",
                    )
                    if center_zero:
                        abs_max = max(slice_vmax, -slice_vmin)
                        slice_vmin, slice_vmax = -abs_max, abs_max

                    logger.debug(f"Inferring (vmin, vmax) = ({slice_vmin}, {slice_vmax})")

                ref = ax.pcolormesh(
                    getattr(self.xi, grid_hu["units"]),
                    getattr(self.eta, grid_hu["units"]),
                    map_slice_data,
                    cmap=cmap,
                    vmin=vmin or slice_vmin,
                    vmax=vmax or slice_vmax,
                )

                ax.set_aspect("equal")

                cbar = fig.colorbar(
                    ref,
                    ax=ax,
                    shrink=0.9,
                    aspect=16,
                    location="right",
                    pad=0.01,
                )

                label = rf"${u['math_name']}$"
                if "stokes" in ax_slices:
                    label += f" Stokes {stokes}"
                if "nu" in ax_slices:
                    label += f" at {self.nu[ax_slices['nu']]}"
                cbar.set_label(label, fontsize=10)

                ax.tick_params(axis="x", bottom=True, top=False)
                ax.tick_params(axis="y", left=True, right=False, rotation=90)

                # lon = ax.coords[0]
                # lon.set_axislabel(rf"{self.frame.phi_long_name}")

                # lat = ax.coords[1]
                # lat.set_axislabel(rf"{self.frame.theta_long_name}")

                ax.set_xlabel(rf"{self.frame.phi_long_name}")
                ax.set_ylabel(rf"{self.frame.theta_long_name}")

                ax2 = ax.secondary_xaxis("top")
                ax2.set_xlabel(rf"$\Delta\,\theta$ [{grid_hu['units']}]")

        if filename is not None:
            plt.savefig(fname=filename, dpi=256)

    def to_fits(self, path: str):

        # if self.dims.get("nu", 0) > 1:
        #     logger.warning("Writing a multifrequency maps ")

        fits.writeto(
            filename=path,
            data=self.data,
            header=self.header(),
            overwrite=True,
        )

    def to_hdf(self, filename: str, compress: bool = True):
        compression_kwargs = {"compression": "gzip", "compression_opts": 9} if compress else {}

        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=np.float32, data=self.data, **compression_kwargs)
            if not (self.weight == 1).all().compute():
                f.create_dataset("weight", dtype=np.float32, data=self.weight, **compression_kwargs)
            if "stokes" in self.dims:
                f.create_dataset("stokes", data=self.stokes, dtype=h5py.string_dtype())
            if "nu" in self.dims:
                f.create_dataset("nu", data=self.nu.Hz)
            if "t" in self.dims:
                f.create_dataset("t", data=self.t.s)
            if "z" in self.dims:
                f.create_dataset("z", data=self.z)
            f.create_dataset("center", dtype=float, data=(self.center[0].deg, self.center[1].deg))
            f.create_dataset("eta", dtype=float, data=self.eta.deg)
            f.create_dataset("xi", dtype=float, data=self.xi.deg)
            f.create_dataset("units", data=self.units)
            f.create_dataset("frame", data=self.frame.name)
            f.create_dataset("beam", data=self.beam.deg)
