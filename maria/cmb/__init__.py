from __future__ import annotations

import logging

import healpy as hp
import numpy as np
import pandas as pd

from ..constants import T_CMB
from ..io import download_from_url, fetch
from ..map import HEALPixMap

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


DEFAULT_CMB_KWARGS = {"nside": 1024}


class CMB:
    def __init__(self, data, fields):
        if len(data) != len(fields):
            raise ValueError("Data and labels must have the same shape!")

        self.data = {}
        for field, M in zip(fields, data):
            self.data[field] = M

        self.nside = int(np.sqrt(len(M) / 12))

    def __getattr__(self, attr):
        if attr in self.data:
            return self.data[attr]
        raise AttributeError(f"No attribute named '{attr}'.")

    @property
    def fields(self):
        return list(self.data.keys())

    def plot(self, field=None, units="uK_CMB"):
        field = field or self.fields[0]
        m = self.data[field]
        vmin, vmax = 1e6 * np.quantile(m[~np.isnan(m)], q=[0.001, 0.999])
        hp.visufunc.mollview(
            1e6 * m,
            min=vmin,
            max=vmax,
            cmap="cmb",
            unit=r"uK$_{CMB}$",
        )


def generate_cmb(nside=2048, seed=123456, **kwargs):
    """
    Generate a new CMB.

    Taken from https://www.zonca.dev/posts/2020-09-30-planck-spectra-healpy.html
    """

    cmb_spectrum_path = fetch(
        "cmb/spectra/planck.csv",
        max_age=30 * 86400,
        # refresh=kwargs.get("refresh_cache", False),
    )

    # in uK
    cl = pd.read_csv(cmb_spectrum_path, index_col=0)
    lmax = cl.index.max()

    alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
    cmb_data = hp.alm2map(alm, nside=nside, lmax=lmax)

    return HEALPixMap(
        data=1e6 * (T_CMB + cmb_data[:, None, None, :]),
        stokes=["I", "Q", "U"],
        units="uK_b",
        nu=150,
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
