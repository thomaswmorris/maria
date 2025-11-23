from __future__ import annotations

import logging

import healpy as hp
import numpy as np
import pandas as pd
import scipy as sp

from ..io import fetch
from ..map import ProjectionMap
from ..units import Quantity
from .cmb import CMB

# shut up healpy I don't care about the resolution
logging.getLogger("healpy").setLevel(logging.WARNING)

CMB_SPECTRUM_CACHE_PATH = "/tmp/maria-data/cmb/spectrum.txt"
CMB_SPECTRUM_CACHE_MAX_AGE = 30 * 86400  # one month

CMB_SOURCES = {
    "planck": {"spectrum": "cmb/spectra/planck.csv"},
    "camb": {"spectrum": "cmb/spectra/camb.csv"},
}


def get_cmb_spectrum(source="camb"):
    cmb_spectrum_path = fetch(
        CMB_SOURCES[source]["spectrum"],
        max_age=CMB_SPECTRUM_CACHE_MAX_AGE,
    )

    return pd.read_csv(cmb_spectrum_path, index_col=0)


def generate_cmb(nside: int = 2048, seed=123456, **kwargs):
    """
    Generate a new CMB.

    Taken from https://www.zonca.dev/posts/2020-09-30-planck-spectra-healpy.html
    """

    cl = get_cmb_spectrum(source="planck")
    lmax = cl.ell.max()

    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    cmb_data = hp.alm2map(alm, nside=nside, lmax=lmax)

    return CMB(
        data=cmb_data[:, None, None, :],
        stokes="IQU",
        units="K_CMB",
        frame="galactic",
        nu=148e9,
    )


def generate_cmb_patch(
    width: float = 5.0,
    height: float = None,
    center: tuple[float, float] = None,
    resolution: float = None,
    frame: str = "ra/dec",
    degrees: bool = True,
    buffer: int = 2,
):
    width = Quantity(width, "deg" if degrees else "rad").rad
    height = Quantity(height, "deg" if degrees else "rad").rad if height is not None else width
    resolution = Quantity(resolution, "deg" if degrees else "rad").rad if resolution is not None else width / 256
    center = Quantity(center, "deg" if degrees else "rad").rad if center is not None else (0.0, 0.0)

    nx = int(width / resolution)
    ny = int(height / resolution)

    dx = width / nx
    dy = height / ny

    x_buffer_pixels = buffer * nx
    y_buffer_pixels = buffer * ny

    nx_gen = 2 * x_buffer_pixels + nx
    ny_gen = 2 * y_buffer_pixels + ny

    kx = np.fft.fftfreq(nx_gen, dx)
    ky = np.fft.fftfreq(ny_gen, dy)
    KY, KX = np.meshgrid(ky, kx)
    K = np.sqrt(KX**2 + KY**2)

    cmb_spectrum = get_cmb_spectrum()
    PS = sp.interpolate.interp1d(cmb_spectrum.ell.values, cmb_spectrum.TT.values, bounds_error=False, fill_value=0)(
        2 * np.pi * K
    )

    PS /= resolution**2  # convert from uK^2 radians^2 to uK^2 pixels^2

    complex_map = np.fft.ifft2(np.sqrt(PS) * np.fft.fft2(np.random.standard_normal((ny_gen, nx_gen))))
    m = complex_map.real[:nx, :ny]

    return ProjectionMap(
        data=(m - m.mean())[None], center=center, width=width, nu=148e9, frame=frame, units="K_CMB", degrees=False
    )
