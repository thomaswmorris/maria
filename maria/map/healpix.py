import os

import dask.array as da
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from ..units import Quantity, parse_units
from .base import Map

here, this_filename = os.path.split(__file__)


class HEALPixMap(Map):
    """
    A spherical map. It has shape (stokes, nu, t, pix).
    """

    def __init__(
        self,
        data: float,
        weight: float = None,
        stokes: float = None,
        nu: float = None,
        t: float = None,
        frame: str = "ra_dec",
        units: str = "K_RJ",
    ):
        # give it four dimensions
        data = data * np.ones((1, 1, 1, 1))

        super().__init__(data=data, weight=weight, stokes=stokes, nu=nu, t=t, units=units)

        self.frame = frame

        parse_units(units)

        self.units = units
        self.npix = self.data.shape[-1]

        if not hp.pixelfunc.isnpixok(self.npix):
            raise ValueError(f"Invalid pixel count (n={self.npix}).")

        self.nside = hp.pixelfunc.npix2nside(self.npix)

        if len(self.nu) != self.n_nu:
            raise ValueError(
                f"Number of supplied frequencies ({len(self.nu)}) does not match the "
                f"nu dimension of the supplied map ({self.n_nu}).",
            )

    def __repr__(self):
        parts = []

        parts.append(
            f"shape[stokes, nu, t, pix]=({self.n_stokes}, {self.n_nu}, {self.n_t}, {self.npix})",
        )
        parts.append(f"nside={self.nside}")

        return f"HEALPixMap({', '.join(parts)})"

    @property
    def resolution(self):
        return hp.pixelfunc.nside2resol(self.nside)

    def package(self):
        return {k: getattr(self, k) for k in ["data", "weight", "stokes", "nu", "t", "frame", "units"]}

    @property
    def X(self):
        return np.meshgrid(self.x_side, self.y_side)[0]

    @property
    def Y(self):
        return np.meshgrid(self.x_side, self.y_side)[1]

    def smooth(self, sigma: float = None, fwhm: float = None, inplace: bool = False):
        if not (sigma is None) ^ (fwhm is None):
            raise ValueError("You must supply exactly one of 'sigma' or 'fwhm'.")

        sigma = sigma if sigma is not None else fwhm / np.sqrt(8 * np.log(2))

        data = np.stack([hp.sphtfunc.smoothing(m, sigma=sigma) for m in self.data.reshape(-1, self.npix)]).reshape(
            self.data.shape
        )

        if inplace:
            self.data = data

        else:
            package = self.package()
            package.update({"data": data})
            return type(self)(**package)

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", dtype=float, data=self.data)

            if self._weight is not None:
                f.create_dataset("weight", dtype=float, data=self._weight)

            for field in ["nu", "t", "frame", "units"]:
                f.create_dataset(field, data=getattr(self, field))

    def plot(self):
        for i in range(len(self.stokes)):
            m = self.data[i, 0, 0]
            min, max = da.percentile(m, q=[0.1, 99.9])

            hp.newvisufunc.projview(
                m.compute(),
                min=min,
                max=max,
                cmap="cmb",
                sub=(len(self.stokes), 1, 1 + i),
            )
            # hp.visufunc.gnomview(self.data[i,0,0], cmap="cmb")

        plt.tight_layout()

    def __repr__(self):
        return f"""{self.__class__.__name__}:
  nside: {self.nside}
  stokes: {self.stokes}
  nu: {Quantity(self.nu, "Hz")}
  t: {Quantity(self.t, "s")}
  quantity: {self.u["quantity"]}
  units: {self.units}
  resolution: {Quantity(self.resolution, "rad")}"""
