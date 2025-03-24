from __future__ import annotations

import copy
import json
import logging
import time as ttime

import arrow
import h5py
import numpy as np
import scipy as sp
from astropy.io import fits
from dask import array as da

from ..array import ArrayList
from ..atmosphere import AtmosphericSpectrum
from ..coords import Coordinates
from ..io import DEFAULT_TIME_FORMAT, humanize_time
from ..plotting import tod_plot, twinkle_plot

logger = logging.getLogger("maria")


class TOD:
    """
    Time-ordered data. This has per-detector pointing and data.
    """

    def __init__(
        self,
        data: dict,
        weight: float = None,
        coords: Coordinates = None,
        units: str = "K_RJ",
        dets: ArrayList = None,
        dtype: type = np.float32,
        metadata: dict = {},
    ):
        self.weight = weight
        self.coords = coords
        self.dets = dets
        self.header = fits.header.Header()
        self.units = units
        self.dtype = dtype
        self.metadata = metadata

        self.data = {}

        for field, field_data in data.items():
            if field_data.ndim != 2:
                raise ValueError("Only two-dimensional TODs are currently supported.")

            self.data[field] = da.asarray(field_data)

        # sort them alphabetically
        self.data = {k: self.data[k] for k in sorted(list(self.fields))}

        if self.weight is None:
            self.weight = da.ones_like(self.signal)

    @property
    def spectrum(self):
        if not hasattr(self, "_spectrum"):
            if "region" in self.metadata:
                self._spectrum = AtmosphericSpectrum(self.metadata["region"])
            else:
                self._spectrum = None
        return self._spectrum

    @property
    def boresight(self):
        if not hasattr(self, "_boresight"):
            self._boresight = self.coords.boresight()
        return self._boresight

    def calibration_kwargs(self, band=None):
        kwargs = {"elevation": self.el[self.dets.band_name == band.name] if band else self.el}

        if self.metadata["atmosphere"]:
            kwargs["spectrum"] = self.spectrum
            kwargs["zenith_pwv"] = self.metadata["pwv"]
            kwargs["base_temperature"] = self.metadata["base_temperature"]

        else:
            kwargs["spectrum"] = None

        return kwargs

    def to(self, units: str):
        """
        Convert to a different set of units.
        """

        cal_start_s = ttime.monotonic()

        # make sure that all Array have a band that the TOD knows about
        for band_name in np.unique(self.dets.band_name):
            if band_name not in self.dets.bands.name:
                raise ValueError(f"No band defined for detector with band '{band_name}'.")

        content = self.content
        for band in self.dets.bands:
            band_mask = self.dets.band_name == band.name

            if band_mask.sum() == 0:
                continue

            # this is to handle transmission

            cal = band.cal(f"{self.units} -> {units}", **self.calibration_kwargs(band))

            for field in self.fields:
                content["data"][field][band_mask] = cal(self.data[field][band_mask])

        content["units"] = units

        logger.debug(f'Converted {self} to units "{units}" in {humanize_time(ttime.monotonic() - cal_start_s)}.')

        return TOD(**content)

    @property
    def shape(self):
        return self.signal.shape

    @property
    def fields(self) -> list:
        return sorted(list(self.data.keys()))

    @property
    def signal(self) -> da.Array:
        return sum([self.data[field] for field in self.fields])

    @property
    def duration(self) -> float:
        return np.ptp(self.time)

    @property
    def dt(self) -> float:
        return float(np.gradient(self.time, axis=-1).mean())

    @property
    def sample_rate(self) -> float:
        return float(1 / self.dt)

    @property
    def fs(self) -> float:
        return self.sample_rate

    @property
    def nd(self) -> int:
        return int(self.signal.shape[0])

    @property
    def nt(self) -> int:
        return int(self.signal.shape[-1])

    @property
    def start(self):
        return arrow.get(self.time.max())

    @property
    def end(self):
        return arrow.get(self.time.max())

    def subset(
        self,
        det_mask: bool | int = None,
        time_mask: bool | int = None,
        band: str = None,
        fields: list = None,
    ):
        det_mask = det_mask or np.arange(self.nd)
        time_mask = time_mask or np.arange(self.nt)
        fields = fields or self.fields

        if band is not None:
            det_mask = self.dets.band_name == band
            if not det_mask.sum() > 0:
                raise ValueError(f"There are no Array for band '{band}'.")

        if time_mask is not None:
            if len(time_mask) != self.nt:
                raise ValueError("The detector mask must have shape (n_dets,).")

        if det_mask is not None:
            if not (len(det_mask) == self.nd):
                raise ValueError("The detector mask must have shape (n_dets,).")

        content = self.content
        content.update(
            {
                "data": {field: self.data[field][det_mask][..., time_mask] for field in fields},
                "weight": self.weight[det_mask][..., time_mask],
                "coords": self.coords[det_mask][..., time_mask],
                "dets": self.dets._subset(det_mask),
            }
        )

        return TOD(**content)

    @property
    def time(self):
        return self.coords.t

    @property
    def earth_location(self):
        return self.coords.earth_location

    @property
    def lat(self):
        return np.round(self.earth_location.lat.deg, 6)

    @property
    def lon(self):
        return np.round(self.earth_location.lon.deg, 6)

    @property
    def alt(self):
        return np.round(self.earth_location.height.value, 6)

    @property
    def az(self):
        return self.coords.az

    @property
    def el(self):
        return self.coords.el

    @property
    def ra(self):
        return self.coords.ra

    @property
    def dec(self):
        return self.coords.dec

    @property
    def azim_phase(self):
        return np.pi * (sp.signal.sawtooth(2 * np.pi * self.time / self.azim_scan_period, width=1) + 1)

    @property
    def turnarounds(self):
        azim_grad = sp.ndimage.gaussian_filter(np.gradient(self.azim), sigma=16)
        return np.where(np.sign(azim_grad[:-1]) != np.sign(azim_grad[1:]))[0]

    def splits(self, target_split_time: float = None):
        if target_split_time is None:
            return list(zip(self.turnarounds[:-1], self.turnarounds[1:]))
        else:
            fs = self.fs
            splits_list = []
            for s, e in self.splits(target_split_time=None):
                split_time = self.time[e] - self.time[s]  # total time in the split
                n_splits = int(
                    np.ceil(split_time / target_split_time),
                )  # number of new splits
                n_split_samples = int(
                    target_split_time * fs,
                )  # number of samples per new split
                for split_start in np.linspace(s, e - n_split_samples, n_splits).astype(
                    int,
                ):
                    splits_list.append(
                        (split_start, np.minimum(split_start + n_split_samples, e)),
                    )
            return splits_list

    def to_fits(self, fname, format="MUSTANG-2"):
        """
        Save the TOD to a fits file
        """

        if format.lower() == "mustang-2":
            header = fits.header.Header()

            header["AZIM"] = (self.coords.center("az_el")[0].compute(), "radians")
            header["ELEV"] = (self.coords.center("az_el")[1].compute(), "radians")
            header["BMAJ"] = (8.0, "arcsec")
            header["BMIN"] = (8.0, "arcsec")
            header["BPA"] = (0.0, "degrees")
            header["NDETS"] = self.dets.n

            header["SITELAT"] = (self.lat, "Site Latitude")
            header["SITELONG"] = (self.lon, "Site Longitude")
            header["SITEELEV"] = (self.alt, "Site elevation (meters)")

            col01 = fits.Column(
                name="DX   ",
                format="E",
                array=self.coords.ra.flatten(),
                unit="radians",
            )
            col02 = fits.Column(
                name="DY   ",
                format="E",
                array=self.coords.dec.flatten(),
                unit="radians",
            )

            tod_rj = self.to(units="K_RJ")
            col03 = fits.Column(
                name="FNU  ",
                format="E",
                array=tod_rj.signal.compute().flatten(),
                unit=tod_rj.units,
            )
            col04 = fits.Column(name="UFNU ", format="E")
            col05 = fits.Column(
                name="TIME ",
                format="E",
                array=((self.time - self.time[0]) * np.ones_like(self.coords.ra)).flatten(),
                unit="s",
            )
            col06 = fits.Column(name="COL  ", format="I")
            col07 = fits.Column(name="ROW  ", format="I")
            col08 = fits.Column(
                name="PIXID",
                format="I",
                array=(
                    np.arange(len(self.coords.ra), dtype=np.int16).reshape(-1, 1) * np.ones_like(self.coords.ra)
                ).flatten(),
            )
            col09 = fits.Column(
                name="SCAN ",
                format="I",
                array=np.zeros_like(self.coords.ra).flatten(),
            )
            col10 = fits.Column(name="ELEV ", format="E")

            hdu = fits.BinTableHDU.from_columns(
                [col01, col02, col03, col04, col05, col06, col07, col08, col09, col10],
                header=header,
            )

            hdu.writeto(fname, overwrite=True)

    def to_hdf(self, fname):
        with h5py.File(fname, "w") as f:
            f.createdataset(fname)

    def plot(self, detrend=True, mean=True, n_freq_bins: int = 256):
        tod_plot(
            self,
            detrend=detrend,
            n_freq_bins=n_freq_bins,
        )

    def twinkle(self, filename=None, **kwargs):
        twinkle_plot(
            self,
            filename=filename,
            **kwargs,
        )

    def __getattr__(self, attr):
        if attr in self.fields:
            return self.data[attr]
        raise AttributeError(f"'TOD' object has no attribute '{attr}'")

    def __repr__(self):
        parts = []
        parts.append(f"shape={self.shape}")
        parts.append(f"fields={repr(self.fields)}")
        parts.append(f"units={repr(self.units)}")
        parts.append(f"start={self.start.format(DEFAULT_TIME_FORMAT)}")
        parts.append(f"duration={self.duration:.01f}s")
        parts.append(f"sample_rate={self.sample_rate:.01f}Hz")
        parts.append(f"metadata={self.metadata}")
        return f"TOD({', '.join(parts)})"

    @property
    def content(self):
        res = {"data": {}}
        for field in self.fields:
            res["data"][field] = copy.deepcopy(self.data[field])
        for key in ["coords", "weight", "units", "dets", "dtype", "metadata"]:
            if hasattr(self, key):
                res[key] = getattr(self, key)
        return res

    def copy(self):
        """
        Copy yourself.
        """
        return TOD(**self.content)


def check_nested_keys(keys_found, data, keys):
    for key in data.keys():
        for i in range(len(keys)):
            if keys[i] in data[key].keys():
                keys_found[i] = True


def check_json_file_for_key(keys_found, file_path, *keys_to_check):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return check_nested_keys(keys_found, data, keys_to_check)


def test_multiple_json_files(files_to_test, *keys_to_find):
    keys_found = np.zeros(len(keys_to_find)).astype(bool)

    for file_path in files_to_test:
        check_json_file_for_key(keys_found, file_path, *keys_to_find)

    if np.sum(keys_found) != len(keys_found):
        raise KeyError(np.array(keys_to_find)[~keys_found])
