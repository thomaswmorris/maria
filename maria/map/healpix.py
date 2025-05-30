import os

import dask.array as da
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ..coords import Coordinates, frames
from ..units import Quantity
from ..utils import compute_pointing_matrix
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
        z: float = None,
        beam: tuple[float, float, float] = None,
        frame: str = "ra_dec",
        units: str = "K_RJ",
        degrees: bool = True,
        dtype: type = np.float32,
    ):
        if weight is not None:
            if weight.shape != data.shape:
                raise ValueError(f"'data' {data.shape} and 'weight' {weight.shape} should have the same shape.")
        else:
            weight = da.ones_like(data)

        map_dims = {"npix": data.shape[-1]}

        super().__init__(
            data=data,
            weight=weight,
            stokes=stokes,
            nu=nu,
            t=t,
            z=z,
            beam=beam,
            map_dims=map_dims,
            units=units,
            degrees=degrees,
            dtype=dtype,
        )

        if not hp.pixelfunc.isnpixok(self.npix):
            raise ValueError(f"Invalid pixel count (n={self.npix}).")

        self.nside = hp.pixelfunc.npix2nside(self.npix)
        self.frame = frame

        if not hasattr(beam, "__len__"):
            beam = beam or self.resolution
            beam = (beam, beam, 0)

        if len(beam) != 3:
            raise ValueError("'beam' must be either a number or a tuple of (major, minor, angle)")

        self.beam = tuple(np.radians(beam) if degrees else beam)

    def pointing_matrix(self, coords: Coordinates):
        idx = hp.ang2pix(
            nside=self.nside,
            phi=getattr(coords, frames[self.frame]["phi"]).ravel(),
            theta=np.pi / 2 - getattr(coords, frames[self.frame]["theta"]).ravel(),
        ).ravel()

        nsamps = len(idx)
        return sp.sparse.csc_array(
            (np.ones(coords.size, dtype=np.uint8), (idx, np.arange(nsamps))), shape=(self.npix, nsamps)
        )

    @property
    def resolution(self):
        return hp.pixelfunc.nside2resol(self.nside)

    @property
    def npix(self):
        return self.dims["npix"]

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
  shape{self.dims_string}: {self.data.shape}
  stokes: {self.stokes if "stokes" in self.dims else "naive"}
  nu: {Quantity(self.nu, "Hz") if "nu" in self.dims else "naive"}
  t: {Quantity(self.t, "s") if "t" in self.dims else "naive"}
  nside: {self.nside}
  quantity: {self.u["quantity"]}
  units: {self.units}
    min: {np.nanmin(self.data).compute():.03e}
    max: {np.nanmax(self.data).compute():.03e}
  resolution: {Quantity(self.resolution, "rad")}
  memory: {Quantity(self.data.nbytes + self.weight.nbytes, "B")}"""
