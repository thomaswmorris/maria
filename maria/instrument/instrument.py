from __future__ import annotations

import copy
import glob
import json
import os
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from ..array import Array, ArrayList, get_array_config  # noqa
from ..array.array import DET_COLUMN_UNITS
from ..band import BAND_CONFIGS, Band, BandList, parse_band  # noqa
from ..beam import compute_angular_fwhm
from ..io import flatten_config, read_yaml
from ..units import Quantity
from ..utils import HEX_CODE_LIST

subarray_params_to_inherit = [
    "array_packing",
    "array_shape",
    "array_offset",
    "beam_spacing",
    "max_baseline",
    "baseline_packing",
    "baseline_shape",
    "baseline_offset",
    "bath_temp",
    "polarization",
    "primary_size",
    "bands",
    "field_of_view",
]

band_params_to_inherit = [
    "time_constant",  # seconds
    "white_noise",  # Kelvin
    "pink_noise",  # Kelvin / s
    "efficiency",
]

passband_params_to_inherit = {
    "center": 150,  # GHz
    "width": 30,  # GHz
    "shape": "top_hat",
}

BAND_COLUMN_UNITS = {"NEP": "W√s", "NEP_per_loading": "W√s", "knee": "Hz", "time_constant": "s", "nu": "Hz"}


def fits_units(units):
    # fits units can't have a square root symbol
    return units.replace("√s", " s(1/2)")


allowed_subarray_params = {
    "n": "int",
    "array_packing": "float",
    "array_shape": "float",
    "array_offset": "float",
    "beam_spacing": "float",
    "max_baseline": "float",
    "baseline_packing": "float",
    "baseline_shape": "float",
    "baseline_offset": "float",
    "bath_temp": "float",
    "polarization": "float",
    "primary_size": "float",
    "field_of_view": "float",
    "bands": "float",
}


class Instrument:
    @classmethod
    def from_config(cls, config):
        c = config.copy()

        if "array" in c:
            c["arrays"] = [c.pop("array")]

        for key in ["aliases"]:
            if key in c:
                c.pop(key)

        return cls(**c)

    @classmethod
    def from_fits(cls, fname):
        arrays = []
        with fits.open(fname) as f:
            header = f[0].header
            bands = Table.read(f["BANDS"], character_as_bytes=False, mask_invalid=False)

            for hdu in f[1:]:
                if hdu.name != "DETS":
                    continue

                array_name = hdu.header["ARRNAME"]
                dets = Table.read(hdu, character_as_bytes=False, mask_invalid=False).to_pandas()

                array_bands = []
                for row in bands[bands["array_name"] == array_name]:
                    spectrum_kwargs = json.loads(row["spectrum"])
                    band = Band(
                        name=row["name"],
                        nu=np.array(row["nu"]),
                        tau=np.array(row["tau"]),
                        shape=row["shape"],
                        NEP=row["NEP"],
                        NEP_per_loading=row["NEP_per_loading"],
                        knee=row["knee"],
                        time_constant=row["time_constant"],
                        gain_error=row["gain_error"],
                        spectrum_kwargs={"region": spectrum_kwargs.pop("region")} if spectrum_kwargs else {},
                    )

                    # supplying nu and tau renormalizes them, so put back the exact values
                    band.tau = np.array(row["tau"])
                    band.efficiency = float(row["efficiency"])
                    if spectrum_kwargs:
                        band.spectrum_kwargs = spectrum_kwargs

                    array_bands.append(band)

                arrays.append(Array(name=array_name, dets=dets, bands=array_bands))

        return cls(
            arrays=arrays,
            description=header["DESCRIP"],
            documentation=header["DOCS"],
            az_vel_limit=header["VELLIM"],
            acc_limit=header["ACCLIM"],
        )

    def __init__(
        self,
        arrays: ArrayList | list | dict,
        description: str = "An instrument.",
        documentation: str = "",
        az_vel_limit: float = 1e2,  # in deg/s
        acc_limit: float = 1e2,  # in deg/s^2
    ):
        """
        Parameters
        ----------
        az_vel_limit : type
            The maximum angular speed of the array.
        """

        # if isinstance(arrays, list):
        #     array_index = 1
        #     for array in arrays:
        #         if "name" not in array:
        #             array["name"] = f"array{array_index}"
        #             array_index += 1

        # array_list = []
        # for array in arrays:
        #     if isinstance(array, Mapping):
        #         array_config = get_array_config(**array)
        #         array_list.append(Array.from_config(array_config))
        #     elif isinstance(array, Array):
        #         array_list.append(array)

        # if isinstance(arrays, ArrayList):
        #     self.arrays = arrays.arrays
        # else:
        #     if isinstance(arrays, list):
        #         array_names = [array.name for i in range(len(arrays))]
        #         array_values = arrays

        #     elif isinstance(arrays, dict):
        #         array_names = list(arrays.keys())
        #         array_values = list(arrays.values())
        #     else:
        #         raise ValueError("'arrays' must be a list or a dict.")

        #     self.arrays = []
        #     for array_name, array in zip(array_names, array_values):
        #         if isinstance(array, Array):
        #             array.name = array.name or array_name
        #             self.arrays.append(array)
        #         elif isinstance(array, dict):
        #             if "name" not in array:
        #                 array["name"] = array_name
        #             self.arrays.append(Array.from_config(array))
        #         elif isinstance(array, str):
        #             self.arrays.append(Array.from_kwargs(name=array_name, key=array))

        self.arrays = ArrayList(arrays)
        self.description = description
        self.documentation = documentation
        self.az_vel_limit = az_vel_limit
        self.acc_limit = acc_limit

        # self.primary_size = float(self.dets.primary_size.max())
        # self.field_of_view = np.round(np.degrees(lazy_diameter(self.dets.offsets)), 3)
        # self.baseline = np.round(lazy_diameter(self.dets.baselines), 3)

        # if self.field_of_view < 0.5 / 60:
        #     self.units = "arcsec"
        # elif self.field_of_view < 0.5:
        #     self.units = "arcmin"
        # else:
        #     self.units = "degrees"

    def __repr__(self):
        dets = self.dets
        band_summaries = []
        for band in self.bands:
            band_dets = dets(band=band.name)
            # detectors can have their own sensitivities, so we show those of the median detector
            median_band = copy.copy(band)
            median_band.NEP = Quantity(np.median(band_dets.NEP), "W√s")
            band_summaries.append({**median_band.summary(), "FWHM": Quantity(band_dets.fwhm.mean(), "rad")})
        band_summary = pd.DataFrame(band_summaries)

        arrays_repr = "\n".join([f"│  {s}" for s in str(self.arrays.summary().__repr__()).split("\n")])
        bands_repr = "\n".join([f"   {s}" for s in str(band_summary.__repr__()).split("\n")])

        s = f"""Instrument({len(self.arrays)} array{"s" if len(self.arrays) > 1 else ""})
├ arrays:
{arrays_repr}
│ 
└ bands:
{bands_repr}"""

        return s

    @property
    def dets(self):
        return self.arrays.combine()

    @property
    def bands(self):
        return self.arrays.bands

    @property
    def xi(self):
        return self.dets.xi

    @property
    def eta(self):
        return self.dets.eta

    @property
    def offsets(self):
        return np.c_[self.xi, self.eta]

    @property
    def baseline_x(self):
        return self.dets.baseline_x

    @property
    def baseline_y(self):
        return self.dets.baseline_y

    @property
    def baseline_z(self):
        return self.dets.baseline_z

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y, self.baseline_z]

    @staticmethod
    def beam_profile(r, fwhm):
        return np.exp(np.log(0.5) * np.abs(r / fwhm) ** 8)

    @property
    def n(self):
        return self.dets.n

    # def angular_beam_filter(self, z, res, beam_profile=None, buffer=1):  # noqa F401
    #     """
    #     Angular beam width (in radians) as a function of depth (in meters)
    #     """
    #     return construct_beam_filter(self.angular_fwhm(z), res, beam_profile=beam_profile, buffer=buffer)

    # def physical_beam_filter(self, z, res, beam_profile=None, buffer=1):  # noqa F401
    #     """
    #     Angular beam width (in radians) as a function of depth (in meters)
    #     """
    #     return construct_beam_filter(self.physical_fwhm(z), res, beam_profile=beam_profile, buffer=buffer)

    def plot(self, z=np.inf, plot_gammas=False):
        self.dets.plot(z=z, plot_gammas=plot_gammas)

    def to_fits(self, fname, overwrite=False):
        header = fits.header.Header()
        header["DESCRIP"] = (self.description, "Instrument description")
        header["DOCS"] = (self.documentation, "Instrument documentation")
        header["VELLIM"] = (self.az_vel_limit, "Azimuth velocity limit (deg/s)")
        header["ACCLIM"] = (self.acc_limit, "Acceleration limit (deg/s^2)")
        header["NARRAYS"] = len(self.arrays)

        hdus = [fits.PrimaryHDU(header=header)]

        band_rows = []
        for array_index, array in enumerate(self.arrays):
            dets = Table.from_pandas(array.dets)
            for col in dets.columns:
                if col in DET_COLUMN_UNITS:
                    dets[col].unit = fits_units(DET_COLUMN_UNITS[col])

            dets_hdu = fits.table_to_hdu(dets)
            dets_hdu.header["EXTNAME"] = "DETS"
            dets_hdu.header["EXTVER"] = array_index + 1
            dets_hdu.header["ARRNAME"] = array.name
            hdus.append(dets_hdu)

            for band in array.bands:
                spectrum_kwargs = {"region": band.spectrum.region, **band.spectrum_kwargs} if band.spectrum else {}
                band_rows.append(
                    {
                        "array_name": array.name,
                        "name": band.name,
                        "shape": band.shape,
                        "efficiency": float(band.efficiency),
                        "NEP": float(band.NEP.to("W√s")),
                        "NEP_per_loading": float(band.NEP_per_loading.to("W√s")),
                        "knee": float(band.knee),
                        "time_constant": float(band.time_constant),
                        "gain_error": float(band.gain_error),
                        "nu": band.nu.Hz,
                        "tau": band.tau,
                        "spectrum": json.dumps(spectrum_kwargs, default=float),
                    }
                )

        bands = Table(rows=band_rows)
        for col, units in BAND_COLUMN_UNITS.items():
            bands[col].unit = fits_units(units)

        bands_hdu = fits.table_to_hdu(bands)
        bands_hdu.header["EXTNAME"] = "BANDS"
        hdus.append(bands_hdu)

        fits.HDUList(hdus).writeto(fname, overwrite=overwrite)
