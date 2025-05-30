from __future__ import annotations

import logging

import healpy as hp
import numpy as np
import pandas as pd

from ..io import download_from_url, fetch
from .cmb import CMB

# shut up healpy I don't care about the resolution
logging.getLogger("healpy").setLevel(logging.WARNING)

CMB_SPECTRUM_SOURCE_URL = (
    "https://github.com/thomaswmorris/maria-data/raw/master/cmb/spectra/"
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
)
CMB_SPECTRUM_CACHE_PATH = "/tmp/maria-data/cmb/spectrum.txt"
CMB_SPECTRUM_CACHE_MAX_AGE = 30 * 86400  # one month

CMB_MAP_SOURCE_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/"
    "COM_CMB_IQU-143-fgsub-sevem_2048_R3.00_full.fits"
)
CMB_MAP_CACHE_PATH = "/tmp/maria-data/cmb/planck/map.fits"
CMB_MAP_CACHE_MAX_AGE = 30 * 86400  # one month


DEFAULT_CMB_KWARGS = {"nside": 2048}


def get_cmb_spectrum():
    cmb_spectrum_path = fetch(
        "cmb/spectra/planck.csv",
        max_age=30 * 86400,
    )

    return pd.read_csv(cmb_spectrum_path, index_col=0)


def generate_cmb(nside=2048, seed=123456, **kwargs):
    """
    Generate a new CMB.

    Taken from https://www.zonca.dev/posts/2020-09-30-planck-spectra-healpy.html
    """

    cl = get_cmb_spectrum()
    lmax = cl.index.max()

    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    cmb_data = hp.alm2map(alm, nside=nside, lmax=lmax)

    return CMB(
        data=cmb_data[:, None, None, :],
        stokes="IQU",
        units="K_CMB",
        frame="galactic",
        nu=148e9,
    )


def get_cmb(**kwargs):
    download_from_url(
        source_url=CMB_MAP_SOURCE_URL,
        cache_path=CMB_MAP_CACHE_PATH,
    )

    field_dtypes = {
        "T": np.float32,
        "Q": np.float32,
        "U": np.float32,
        "T_mask": bool,
        "P_mask": bool,
    }

    maps = {
        field: hp.fitsfunc.read_map(CMB_MAP_CACHE_PATH, field=i).astype(dtype)
        for i, (field, dtype) in enumerate(field_dtypes.items())
    }

    maps["T"] = np.where(maps["T_MASK"], maps["T"], np.nan)
    maps["Q"] = np.where(maps["P_MASK"], maps["Q"], np.nan)
    maps["U"] = np.where(maps["P_MASK"], maps["U"], np.nan)

    return CMB(data=[maps["T"], maps["Q"], maps["U"]], fields=["T", "Q", "U"])
