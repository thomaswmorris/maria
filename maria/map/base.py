import logging
import os
import time as ttime
from collections.abc import Iterable, Mapping

import arrow
import dask.array as da
import numpy as np
import scipy as sp

from ..calibration import Calibration
from ..constants import MARIA_MAX_NU_HZ, MARIA_MIN_NU_HZ
from ..coords import Frame
from ..errors import FrequencyOutOfBoundsError, ShapeError
from ..io import leftpad
from ..units import Quantity, parse_units
from ..utils import compute_resolution_precision

logger = logging.getLogger("maria")

here, this_filename = os.path.split(__file__)

SLICE_DIMS = {
    "stokes": {
        "dtype": str,
        "default": "I",
    },
    "nu": {
        "dtype": float,
        "default": 150e9,
    },
    "t": {
        "dtype": float,
        "default": arrow.now().timestamp(),
    },
    "z": {
        "dtype": float,
        "default": 0,
    },
}

VALID_MAP_QUANTITIES = [
    "rayleigh_jeans_temperature",
    "cmb_temperature_anisotropy",
    "spectral_flux_density_per_pixel",
    "spectral_flux_density_per_beam",
    "spectral_radiance",
    "compton_y",
]


class Map:
    """
    The base class for maps. Maps have data
    """

    def __init__(
        self,
        data: float,
        weight: float,
        stokes: str,
        nu: Iterable[float],
        t: Iterable[float],
        v: Iterable[float],
        z: Iterable[float],
        beam: tuple[float, float, float],  # noqa
        map_dims: dict,
        units: str = "K_RJ",
        frame: str = "ra/dec",
        degrees: bool = True,  # noqa
        dtype: type = np.float32,
    ):
        # check that map units are valid
        u = parse_units(units)

        if u["physical_quantity"] not in VALID_MAP_QUANTITIES:
            raise ValueError(
                f"Passed units '{units}' (with dimension {u['base_units']}) are not valid map units. "
                f"Acceptable map units have the same dimension as one of {VALID_MAP_QUANTITIES}"
            )

        self.units = u["units"]
        self.frame = Frame(frame)

        self.data = da.asarray(data).astype(dtype).squeeze()
        self._weight = da.asarray(weight).squeeze().astype(dtype) if weight is not None else None

        self.map_dims = map_dims

        self.slice_dims = {}
        if stokes is not None:
            stokes = "".join(list(stokes))
            if stokes not in ["I", "IQU", "IQUV"]:
                raise ValueError(f"Invalid stokes parameter '{stokes}' (must be either 'I', 'IQU', or 'IQUV').")
            self.stokes = stokes.upper()
            self.slice_dims["stokes"] = len(self.stokes)
        else:
            self.stokes = ""

        if nu is not None:
            self.nu = Quantity(np.atleast_1d(nu.Hz if isinstance(nu, Quantity) else nu), "Hz")
            if self.nu.ndim > 1:
                raise ShapeError("'nu' can be at most one-dimensional")
            bad_freqs = list(self.nu[(self.nu.Hz < MARIA_MIN_NU_HZ) | (self.nu.Hz > MARIA_MAX_NU_HZ)])
            if bad_freqs:
                raise FrequencyOutOfBoundsError(bad_freqs)
            self.slice_dims["nu"] = len(self.nu)
        else:
            self.nu = Quantity([], "Hz")

        if v is not None:
            self.v = Quantity(np.atleast_1d(v.to("m/s") if isinstance(v, Quantity) else v), "m/s")
            if self.v.ndim > 1:
                raise ShapeError("'v' can be at most one-dimensional")
            self.slice_dims["v"] = len(self.v)
        else:
            self.v = np.array([])

        if t is not None:
            self.t = Quantity(np.atleast_1d(t.seconds if isinstance(t, Quantity) else t), "seconds")
            if self.t.ndim > 1:
                raise ShapeError("'t' can be at most one-dimensional")
            self.slice_dims["t"] = len(self.t)
        else:
            self.t = np.array([])

        if z is not None:
            self.z = np.atleast_1d(z)
            self.slice_dims["z"] = len(self.z)
        else:
            self.z = np.array([])

        implied_shape = tuple(n for n in self.dims.values() if n > 1)

        if not self.data.shape == implied_shape:
            raise ValueError(f"Dimensions imply a data shape {implied_shape}, but data has shape {self.data.shape}")

        if not self.weight.shape == implied_shape:
            raise ValueError(f"Dimensions imply a data shape {implied_shape}, but data has shape {self.weight.shape}")

        for dim_index, (dim, n) in enumerate(self.slice_dims.items()):
            if n == 1:
                self.data = da.expand_dims(self.data, dim_index)
                if self._weight is not None:
                    self._weight = da.expand_dims(self._weight, dim_index)

        if not hasattr(beam, "__len__"):
            beam = np.array([beam, beam, 0])

        if np.shape(beam)[-1] != 3:
            raise ValueError("'beam' must be either a number or a tuple of (major, minor, angle)")

        self.beam = Quantity(beam * np.ones((*self.slice_dims.values(), 3)), "deg" if degrees else "rad")

        if self.u["physical_quantity"] == "spectral_flux_density_per_beam":
            if not np.all(self.beam_area > 0):
                raise ValueError(
                    f"Map is given in units {self.units}, but specified beam(major, minor, angle) = {beam} has zero area"
                )

    @property
    def weight(self):
        return self._weight if self._weight is not None else da.ones_like(self.data)

    @weight.setter
    def weight(self, value):
        if value is not None:
            self._weight = value

    @property
    def dims(self):
        return {**self.slice_dims, **self.map_dims}

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def shape(self):
        return tuple(self.dims.values())

    @property
    def dims_string(self):
        return f"({', '.join(self.dims.keys())})"

    @property
    def dims_list(self):
        return list(self.dims.keys())

    @property
    def nu_bins(self):
        return np.array([0, *(self.nu.Hz[1:] + self.nu.Hz[:-1]) / 2, np.inf])

    @property
    def nu_side(self):
        return self.nu.Hz

    @property
    def t_bins(self):
        """
        This might break in the year 3000.
        """
        # t_min = arrow.get("0001-01-01").timestamp()
        # t_max = arrow.get("3000-01-01").timestamp()
        # return np.array([-np.inf, *(self.t[1:] + self.t[:-1]) / 2, np.inf])

        return np.array([-np.inf, *(self.t[1:] + self.t[:-1]) / 2, np.inf])

    @property
    def t_side(self):
        return (self.t_bins[:-1] + self.t_bins[1:]) / 2

    @property
    def u(self):
        return parse_units(self.units)

    # @property
    # def weight(self):
    #     return self._weight if self._weight is not None else da.ones(shape=self.data.shape)

    def append(self, map, dim):
        return concatenate([self, map], dim=dim)

    def extend(self, maps, dim):
        return concatenate([self, *maps], dim=dim)

    def squeeze(self, dims=None):

        dims = np.atleast_1d(dims) if dims is not None else [dim for dim, n in self.slice_dims.items() if n == 1]
        package = self.package()
        dim_indices_to_squeeze = []

        for dim_index, name in enumerate(dims):
            if name in self.map_dims:
                raise ValueError(f"Cannot squeeze dimension '{name}'")

            n = self.slice_dims.get(name)

            if n is None:
                raise ValueError(f"{self.__class__.__name__} has no dimension '{name}'")
            if n != 1:
                raise ValueError(f"Cannot squeeze dimension '{name}' with length {n} > 1")

            dim_indices_to_squeeze.append(dim_index)
            package.pop(name)

        package["data"] = package["data"].squeeze(tuple(dim_indices_to_squeeze))
        package["weight"] = package["weight"].squeeze(tuple(dim_indices_to_squeeze))

        return type(self)(**package)

    def unsqueeze(self, dim, value=None):
        if dim not in SLICE_DIMS:
            raise ValueError(f"'{dim}' is not a valid map dimension")
        if dim in self.dims:
            raise Exception(f"{self.__class__.__name__} already has dimension '{dim}'")

        new_dim_index = 0
        for d in ["stokes", "nu", "t", "z"]:
            if d == dim:
                break
            if d in self.dims:
                new_dim_index += 1

        package = self.package()
        package["data"] = np.expand_dims(package["data"], new_dim_index)
        package["weight"] = np.expand_dims(package["weight"], new_dim_index)
        package["beam"] = np.expand_dims(package["beam"], new_dim_index)

        if value is None:
            value = SLICE_DIMS[dim]["default"]
        package[dim] = value

        return type(self)(**package)

    @property
    def beam_area(self):
        """
        Returns the beam area in steradians
        """
        area = (np.pi / 4) * self.beam[..., 0].radians * self.beam[..., 1].radians
        area = np.expand_dims(area, axis=(-1, -2)[: len(self.map_dims)])
        return Quantity(area, "sr")

    def beam_repr(self):
        slice_axes = tuple(range(self.beam.ndim - 1))
        if any(self.beam.std(axis=slice_axes) > 0):
            return "ragged"
        b = self.beam.mean(axis=slice_axes)
        return (b[0], b[1], b[2])

    def to(self, units: str, only_return_data: bool = False, **calibration_kwargs: Mapping):
        if units == self.units:
            return self

        u = parse_units(units)

        if u["physical_quantity"] not in VALID_MAP_QUANTITIES:
            raise ValueError(
                f"Units '{units}' (with associated physical quantity '{u['physical_quantity']}') are not valid map units"
            )

        package = self.package().copy()

        # this is just a scaling by some factor
        if u["physical_quantity"] == self.u["physical_quantity"]:
            package["data"] *= self.u["base_units_factor"] / u["base_units_factor"]

        else:
            if "nu" not in self.dims:
                raise ValueError(
                    f"Cannot convert from quantity {self.u['physical_quantity']} to quantity {u['physical_quantity']} "
                    f"when map has no frequency."
                )

            # data = package["data"].swapaxes(0, self.dims_list.index("nu"))  # put the nu index in front
            for nu_index, nu in enumerate(self.nu.Hz):
                nu_key = tuple([nu_index if dim == "nu" else slice(None, None, None) for dim in self.slice_dims])

                cal = Calibration(
                    f"{self.units} -> {units}",
                    nu=nu,
                    pixel_area=self.pixel_area.sr,
                    beam_area=self.beam_area[nu_key].sr,
                    **calibration_kwargs,
                )
                package["data"][nu_key] = cal(package["data"][nu_key])

            # package["data"] = data.swapaxes(0, self.dims_list.index("nu"))  # swap the axes back

        if only_return_data:
            return package["data"]

        package["units"] = units
        return type(self)(**package)

    def sample_nu(self, nu):
        map_nu_interpolator = sp.interpolate.interp1d(self.nu.Hz, self.data, axis=1, kind="linear")

        nu_maps = []
        for nu in np.atleast_1d(nu):
            if nu < self.nu.Hz[0]:
                nu_maps.append(self.data[:, 0])
            elif not (nu < self.nu.Hz[-1]):  # this will include nan
                nu_maps.append(self.data[:, -1])
            else:
                nu_maps.append(map_nu_interpolator(nu))

        return np.stack(nu_maps, axis=1)

    @property
    def nu_bin_bounds(self):
        nu_boundaries = [0, *(self.nu.Hz[:-1] + self.nu.Hz[1:]) / 2, np.inf]
        return [(Quantity(nu1, "Hz"), Quantity(nu2, "Hz")) for nu1, nu2 in zip(nu_boundaries[:-1], nu_boundaries[1:])]

    def copy(self):
        return type(self)(**self.package().copy())

    def compute_stats(self):
        d = np.where(np.isfinite(self.data), self.data, 0)
        w = np.where(np.isfinite(self.data), self.weight, 0)
        md = np.sum(d * w) / np.sum(w)

        self._stats = {
            "min": np.min(d).compute(),
            "max": np.max(d).compute(),
            "rms": np.sqrt(np.sum(np.square(d - md) * w / w.sum())).compute(),
        }

    def __getattr__(self, attr):
        broadcasted_attrs = ["STOKES", "NU", "T", "Y", "X"]
        if attr in broadcasted_attrs:
            broadcasted_attr_values = np.meshgrid(self.stokes, self.nu.Hz, self.t, self.eta, self.xi)
            return broadcasted_attr_values[broadcasted_attrs.index(attr)]
        if attr in ["min", "max", "rms"]:
            if not hasattr(self, "_stats"):
                self.compute_stats()
            return self._stats[attr]

        raise AttributeError(f"{type(self)} object has no attribute '{attr}'")

    @staticmethod
    def __repr_stokes__(stokes):
        return f"stokes({len(stokes)}):\n  components: {stokes}"

    @staticmethod
    def __repr_unitful_range__(x, name):
        if x.size < 4:
            return f"{name}({len(x)}):\n  values: {x}"

        dnu = np.gradient(x.human_value)
        res = "irregular" if np.std(dnu) / np.ptp(x.human_value) > 1e-6 else Quantity(np.mean(dnu), x.human_units)
        prec = compute_resolution_precision(np.sort(x.human_value)[[0, -1]])
        return f"""{name}({len(x)}):
  min: {x.min().__repr__(prec)}
  max: {x.max().__repr__(prec)}
  res: {res}"""

    @staticmethod
    def __repr_t__(t):
        if t.size < 4:
            return f"t({len(t)}):\n  values: {t}"

        t_seconds = t.seconds
        dt = np.gradient(t_seconds)
        res = "irregular" if np.std(dt) / np.ptp(t_seconds) > 1e-4 else Quantity(np.mean(dt), "seconds")
        return f"""t({len(t)}):
  min: {t.min().date}
  max: {t.max().date}
  res: {res}"""

    def __repr_base__(self):
        repr_slice_parts = [
            f"""data{self.data.shape}:
  min: {self.min:.03e}
  max: {self.max:.03e}
  units: {self.units}
  quantity: {self.u["physical_quantity"]}"""
        ]
        if "stokes" in self.dims:
            repr_slice_parts.append(self.__repr_stokes__(self.stokes))
        if "nu" in self.dims:
            repr_slice_parts.append(self.__repr_unitful_range__(self.nu, "nu"))
        if "v" in self.dims:
            repr_slice_parts.append(self.__repr_unitful_range__(self.v, "v"))
        if "z" in self.dims:
            repr_slice_parts.append(f"z({len(self.z)}):\n  values: {self.z}")
        if "t" in self.dims:
            repr_slice_parts.append(self.__repr_t__(self.t))

        return leftpad("\n".join(repr_slice_parts), 2, " ")


def concatenate(maps: list[Map], dim: str) -> Map:
    packages = []
    concat_dim_values = []
    dims_list = {}
    for m in maps:
        for d, n in m.dims.items():
            if d not in dims_list:
                dims_list[d] = []
            dims_list[d].append(n)
        for attr in ["center", "width", "height"]:
            # TODO: checks for healpix maps
            if getattr(m, attr, None) != getattr(maps[0], attr, None):
                print("warning")
        concat_dim_values.extend(getattr(m, dim))
        packages.append(m.to(maps[0].units).package())

    for d, ns in dims_list.items():
        if d != dim:
            if len(set(ns)) > 1:
                raise ShapeError(
                    "Map dimensions must be equal except along the concatenation axis "
                    f"(maps have shapes {[m.shape for m in maps]})."
                )

    dim_index = list(dims_list.keys()).index(dim)
    total_package = packages[0].copy()
    total_package["data"] = np.concatenate([p["data"] for p in packages], axis=dim_index)
    total_package["weight"] = np.concatenate([p["weight"] for p in packages], axis=dim_index)
    total_package[dim] = concat_dim_values

    return type(maps[0])(**total_package)
