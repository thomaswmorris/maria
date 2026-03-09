from __future__ import annotations

import healpy as hp
import numpy as np

from ..io import fetch
from ..map import HEALPixMap

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
CMB_MAP_CACHE_PATH = "cmb/planck.fits"
CMB_MAP_CACHE_MAX_AGE = 30 * 86400  # one month


CMB_SOURCES = {
    "planck": {"spectrum": "cmb/spectra/planck.csv"},
    "camb": {"spectrum": "/Users/tom/maria/data/cmb/spectra/camb.csv"},
}


class CMB(HEALPixMap):
    def __init__(
        self,
        data: float,
        weight: float = None,
        stokes: float = None,
        frame: str = "galactic",
        nu: float = None,
        units: str = "K_CMB",
        **kwargs,
    ):
        super().__init__(data=data, weight=weight, stokes=stokes, nu=nu, z=1100.0, units=units, frame=frame, **kwargs)


def get_cmb(**kwargs):
    cmb_filepath = fetch(source_url=CMB_MAP_SOURCE_URL)
    field_dtypes = {
        "T": np.float32,
        "Q": np.float32,
        "U": np.float32,
        "T_mask": bool,
        "P_mask": bool,
    }

    maps = {
        field: hp.fitsfunc.read_map(cmb_filepath, field=i).astype(dtype)
        for i, (field, dtype) in enumerate(field_dtypes.items())
    }

    maps["T"] = np.where(maps["T_mask"], maps["T"], np.nan)
    maps["Q"] = np.where(maps["P_mask"], maps["Q"], np.nan)
    maps["U"] = np.where(maps["P_mask"], maps["U"], np.nan)

    return CMB(data=np.stack([maps["T"], maps["Q"], maps["U"]], axis=0)[:, None, None], stokes="IQU", nu=143e9)
