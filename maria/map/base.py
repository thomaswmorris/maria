import logging
import os
import time as ttime
from collections.abc import Iterable

import arrow
import dask.array as da
import numpy as np
import scipy as sp

from ..calibration import Calibration
from ..constants import MARIA_MAX_NU, MARIA_MIN_NU
from ..errors import FrequencyOutOfBoundsError
from ..units import Quantity, parse_units

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
}


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
        z: Iterable[float],
        beam: tuple[float, float, float],  # noqa
        map_dims: dict,
        units: str = "K_RJ",
        degrees: bool = True,  # noqa
        dtype: type = np.float32,
    ):
        self.data = da.asarray(data).astype(dtype)
        self.weight = (da.asarray(weight) if weight is not None else da.ones_like(self.data)).astype(dtype)
        self.data, self.weight = np.broadcast_arrays(self.data, self.weight)

        slice_dims = {}
        if stokes is not None:
            if stokes not in ["I", "IQU", "IQUV"]:
                raise ValueError("'stokes' parameter must be either 'I', 'IQU', or 'IQUV'")
            self.stokes = stokes.upper()
            slice_dims["stokes"] = len(self.stokes)
        else:
            self.stokes = ""

        if nu is not None:
            self.nu = np.atleast_1d(nu)
            bad_freqs = list(self.nu[(self.nu < MARIA_MIN_NU) | (self.nu > MARIA_MAX_NU)])
            if bad_freqs:
                raise FrequencyOutOfBoundsError(bad_freqs)
            slice_dims["nu"] = len(self.nu)
        else:
            self.nu = np.array([])

        if t is not None:
            self.t = np.atleast_1d(t)
            slice_dims["t"] = len(self.t)
        else:
            self.t = np.array([])

        if z is not None:
            self.z = np.atleast_1d(z)
            slice_dims["z"] = len(self.z)
        else:
            self.z = np.array([])

        self.dims = {**slice_dims, **map_dims}

        self.data *= np.ones(list(self.dims.values()))
        self.weight *= np.ones(list(self.dims.values()))

        self.units = parse_units(units)["units"]

        # for i, (dim, n) in enumerate(slice_dims.items()):
        #     if self.data.shape[i] != n:
        #         raise ValueError(
        #             f"'{dim}' has length {n} but map has {dim} length {self.data.shape[i]}.",
        #         )

        implied_shape = tuple(list(self.dims.values()))
        # if len(implied_shape) != self.data.ndim:
        #     raise ValueError(f"Inputs imply rank {len(self.dims)} for map data, but data has rank {self.data.ndim}")

        if implied_shape != self.data.shape:
            raise ValueError(
                f"Inputs imply shape {self.dims_string}={implied_shape} for map data, but data has shape {self.data.shape}"
            )

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def dims_string(self):
        return f"({', '.join(self.dims.keys())})"

    @property
    def dims_list(self):
        return list(self.dims.keys())

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
    def u(self):
        return parse_units(self.units)

    # @property
    # def weight(self):
    #     return self._weight if self._weight is not None else da.ones(shape=self.data.shape)

    def squeeze(self, dim):
        package = self.package()
        for i, (name, n) in enumerate(self.dims.items()):
            if (name == dim) or (i == dim):
                if n == 1:
                    package["data"] = package["data"].squeeze(i)
                    package["weight"] = package["weight"].squeeze(i)
                    package.pop(name)
                    return type(self)(**package)
                else:
                    raise ValueError(f"Cannot squeeze dimension '{dim}' with length {n} > 1")
        raise ValueError(f"{self.__class__.__name__} has no dimension '{dim}'")

    def unsqueeze(self, dim, value=None):
        if dim not in SLICE_DIMS:
            raise ValueError(f"'{dim}' is not a valid map dimension")
        if dim in self.dims:
            raise Exception(f"{self.__class__.__name__} already has dimension '{dim}'")

        new_dim_index = 0
        for d in ["stokes", "nu", "t"]:
            if d in self.dims and d != dim:
                new_dim_index += 1

        package = self.package()
        package["data"] = np.expand_dims(package["data"], new_dim_index)
        package["weight"] = np.expand_dims(package["weight"], new_dim_index)
        if value is None:
            value = SLICE_DIMS[dim]["default"]
        package[dim] = value

        return type(self)(**package)

    @property
    def pixel_area(self):
        if hasattr(self, "resolution"):
            return self.resolution**2
        else:
            return self.x_res * self.y_res

    @property
    def beam_area(self):
        return np.pi * self.beam[0] * self.beam[1]

    def to(self, units: str):
        if units == self.units:
            return self

        u = parse_units(units)

        package = self.package().copy()

        # this is just a scaling by some factor
        if u["quantity"] == self.u["quantity"]:
            package["data"] *= self.u["factor"] / u["factor"]

        else:
            if "nu" not in self.dims:
                raise ValueError(
                    f"Cannot convert from quantity {self.u['quantity']} to quantity {u['quantity']} "
                    f"when map has no frequency."
                )

            data = package["data"].swapaxes(0, self.dims_list.index("nu"))  # put the nu index in front
            for nu_index, nu in enumerate(self.nu):
                cal = Calibration(
                    f"{self.units} -> {units}",
                    nu=nu,
                    pixel_area=self.pixel_area,
                    beam_area=self.beam_area,
                )
                data[nu_index] = cal(data[nu_index])

            package["data"] = data.swapaxes(0, self.dims_list.index("nu"))  # swap the axes back

        package["units"] = units
        return type(self)(**package)

    def sample_nu(self, nu):
        map_nu_interpolator = sp.interpolate.interp1d(self.nu, self.data, axis=1, kind="linear")

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
