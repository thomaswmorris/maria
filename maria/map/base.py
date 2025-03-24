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
from ..units import Quantity, parse_units

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

        self.stokes = [param.upper() for param in stokes] if stokes is not None else ["I"]

        self.nu = np.atleast_1d(nu if nu is not None else 150.0e9)

        bad_freqs = list(self.nu[(self.nu < MARIA_MIN_NU) | (self.nu > MARIA_MAX_NU)])
        if bad_freqs:
            qmin_nu = Quantity(MARIA_MIN_NU, units="Hz")
            qmax_nu = Quantity(MARIA_MAX_NU, units="Hz")
            raise ValueError(
                f"Bad frequencies nu={bad_freqs} Hz; maria supports frequencies between "
                f"{qmin_nu.Hz:.0e} ({qmin_nu}) and {qmax_nu.Hz:.0e} ({qmax_nu})."
            )

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
    def u(self):
        return parse_units(self.units)

    @property
    def weight(self):
        return self._weight if self._weight is not None else da.ones(shape=self.data.shape)

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
                if not nu > 0:
                    raise ValueError(f"Cannot convert map with frequency nu={nu}.")

                cal = Calibration(
                    f"{self.units} -> {units}",
                    nu=nu,
                    pixel_area=self.pixel_area,
                )
                data[i] = cal(self.data[i])

        if inplace:
            self.data = data
            self.units = units

        else:
            package = self.package().copy()
            package.update({"data": data, "units": units})
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
