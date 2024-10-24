import functools
import json
import logging

import dask.array as da
import h5py
import matplotlib as mpl  # noqa
import numpy as np
import pandas as pd
import scipy as sp
from astropy.io import fits

from maria import utils

from ..coords import Coordinates
from ..plotting import tod_plot, twinkle_plot

logger = logging.getLogger("maria")


class TOD:
    """
    Time-ordered data. This has per-detector pointing and data.
    """

    def copy(self):
        """
        Copy yourself.
        """
        return TOD(
            coords=self.coords,
            data=self.data,
            units=self.units,
            dets=self.dets,
            abscal=self.abscal,
            dtype=self.dtype,
        )

    def __init__(
        self,
        data,
        weight=None,
        coords: Coordinates = None,
        units: str = "K_RJ",
        dets: pd.DataFrame = None,
        abscal: float = 1.0,
        distributed: bool = True,
        dtype=np.float32,
    ):
        self.weight = weight
        self.coords = coords
        self.dets = dets
        self.header = fits.header.Header()
        self.abscal = abscal
        self.units = units
        self.dtype = dtype

        self.data = {}

        for field, field_data in data.items():
            self.data[field] = field_data
            d = field_data["data"]

            if distributed and not isinstance(d, da.Array):
                self.data[field]["data"] = da.from_array(d)

        # sort them alphabetically
        self.data = {k: self.data[k] for k in sorted(list(self.fields))}

        if self.weight is None:
            self.weight = da.ones_like(self.signal)

    def get_field(self, field: str):
        if field not in self.fields:
            raise ValueError(f"Field '{field}' not found.")

        d = self.data[field]["data"]

        if "scale" in self.data[field]:
            d *= self.data[field]["scale"][..., None]

        if "offset" in self.data[field]:
            d += self.data[field]["offset"][..., None]

        return d

    @property
    def boresight(self):
        if not hasattr(self, "_boresight"):
            self._boresight = self.coords.boresight()
        return self._boresight

    def to(self, units: str):
        cal = self.dets.cal(f"{self.units} -> {units}")

        cal_data = {}
        for field in self.fields:
            cal_data[field] = self.data[field].copy()
            current_scale = cal_data[field].get("scale", 1e0)
            cal_data[field]["scale"] = current_scale * cal

        return TOD(
            coords=self.coords,
            data=cal_data,
            units=units,
            dets=self.dets,
            abscal=self.abscal,
            dtype=self.dtype,
        )

    @property
    def shape(self):
        return self.signal.shape

    @property
    def fields(self) -> list:
        return sorted(list(self.data.keys()))

    @functools.cached_property
    def signal(self) -> da.Array:
        return sum([self.get_field(field) for field in self.fields])

    @property
    def dt(self) -> float:
        return float(np.gradient(self.time, axis=-1).mean())

    @property
    def fs(self) -> float:
        return float(1 / self.dt)

    @property
    def nd(self) -> int:
        return int(self.signal.shape[0])

    @property
    def nt(self) -> int:
        return int(self.signal.shape[-1])

    def subset(
        self, det_mask=None, time_mask=None, band: str = None, fields: list = None
    ):
        fields = fields or self.fields

        if band is not None:
            det_mask = self.dets.band_name == band
            if not det_mask.sum() > 0:
                raise ValueError(f"There are no detectors for band '{band}'.")
            return self.subset(det_mask=det_mask)

        if time_mask is not None:
            if len(time_mask) != self.nt:
                raise ValueError("The detector mask must have shape (n_dets,).")

            subset_coords = Coordinates(
                time=self.coords.time[time_mask],
                phi=self.coords.az[:, time_mask],
                theta=self.coords.el[:, time_mask],
                earth_location=self.earth_location,
                frame="az_el",
            )

            return TOD(
                data={
                    field: data[det_mask]
                    for field, data in self.data.items()
                    if field in fields
                },
                coords=subset_coords,
                dets=self.dets,
                units=self.units,
            )

        if det_mask is not None:
            if not (len(det_mask) == self.nd):
                raise ValueError("The detector mask must have shape (n_dets,).")

            subset_dets = self.dets._subset(det_mask) if self.dets is not None else None

            subset_coords = Coordinates(
                time=self.time,
                phi=self.coords.az[det_mask],
                theta=self.coords.el[det_mask],
                earth_location=self.earth_location,
                frame="az_el",
            )

            return TOD(
                data={
                    field: {k: v[det_mask] for k, v in self.data[field].items()}
                    for field in self.fields
                },
                weight=self.weight[det_mask],
                coords=subset_coords,
                dets=subset_dets,
                units=self.units,
            )

    def process(self, **kwargs):
        D = self.signal.compute()
        W = np.ones(D.shape)

        if "window" in kwargs:
            if "tukey" in kwargs["window"]:
                W *= sp.signal.windows.tukey(
                    D.shape[-1], alpha=kwargs["window"]["tukey"].get("alpha", 0.1)
                )
                D = W * sp.signal.detrend(D, axis=-1)

        if "filter" in kwargs:
            if "window" not in kwargs:
                logger.warning("Filtering without windowing is not recommended.")

            if "f_upper" in kwargs["filter"]:
                D = utils.signal.lowpass(
                    D,
                    fc=kwargs["filter"]["f_upper"],
                    fs=self.fs,
                    order=kwargs["filter"].get("order", 1),
                    method="bessel",
                )

            if "f_lower" in kwargs["filter"]:
                D = utils.signal.highpass(
                    D,
                    fc=kwargs["filter"]["f_lower"],
                    fs=self.fs,
                    order=kwargs["filter"].get("order", 1),
                    method="bessel",
                )

        if "remove_modes" in kwargs:
            n_modes_to_remove = kwargs["remove_modes"]["n"]

            U, V = utils.signal.decompose(
                D, downsample_rate=np.maximum(int(self.fs), 1), mode="uv"
            )
            D = U[:, n_modes_to_remove:] @ V[n_modes_to_remove:]

        if "despline" in kwargs:
            B = utils.signal.get_bspline_basis(
                self.time,
                spacing=kwargs["despline"]["knot_spacing"],
                order=kwargs["despline"].get("spline_order", 3),
            )

            A = np.linalg.inv(B @ B.T) @ B @ D.T
            D -= A.T @ B

        return TOD(
            data={"total": {"data": D}},
            weight=W,
            coords=self.coords,
            units=self.units,
            dets=self.dets,
            dtype=np.float32,
        )

    @property
    def time(self):
        return self.coords.time

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
        return np.pi * (
            sp.signal.sawtooth(2 * np.pi * self.time / self.azim_scan_period, width=1)
            + 1
        )

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
                    np.ceil(split_time / target_split_time)
                )  # number of new splits
                n_split_samples = int(
                    target_split_time * fs
                )  # number of samples per new split
                for split_start in np.linspace(s, e - n_split_samples, n_splits).astype(
                    int
                ):
                    splits_list.append(
                        (split_start, np.minimum(split_start + n_split_samples, e))
                    )
            return splits_list

    def to_fits(self, fname, format="MUSTANG-2"):
        """
        Save the TOD to a fits file
        """

        if format.lower() == "mustang-2":
            header = fits.header.Header()

            header["AZIM"] = (self.coords.center_az, "radians")
            header["ELEV"] = (self.coords.center_el, "radians")
            header["BMAJ"] = (8.0, "arcsec")
            header["BMIN"] = (8.0, "arcsec")
            header["BPA"] = (0.0, "degrees")

            header["SITELAT"] = (self.lat, "Site Latitude")
            header["SITELONG"] = (self.lon, "Site Longitude")
            header["SITEELEV"] = (self.alt, "Site elevation (meters)")

            col01 = fits.Column(
                name="DX   ", format="E", array=self.coords.ra.flatten(), unit="radians"
            )
            col02 = fits.Column(
                name="DY   ",
                format="E",
                array=self.coords.dec.flatten(),
                unit="radians",
            )
            col03 = fits.Column(
                name="FNU  ",
                format="E",
                array=self.signal.flatten(),
                unit=self.units["data"],
            )
            col04 = fits.Column(name="UFNU ", format="E")
            col05 = fits.Column(
                name="TIME ",
                format="E",
                array=(
                    (self.time - self.time[0]) * np.ones_like(self.coords.ra)
                ).flatten(),
                unit="s",
            )
            col06 = fits.Column(name="COL  ", format="I")
            col07 = fits.Column(name="ROW  ", format="I")
            col08 = fits.Column(
                name="PIXID",
                format="I",
                array=(
                    np.arange(len(self.coords.ra), dtype=np.int16).reshape(-1, 1)
                    * np.ones_like(self.coords.ra)
                ).flatten(),
            )
            col09 = fits.Column(
                name="SCAN ", format="I", array=np.zeros_like(self.coords.ra).flatten()
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
            return self.get_field(attr)
        raise AttributeError(f"No attribute named '{attr}'.")

    def __repr__(self):
        return f"TOD(shape={self.shape}, fields={self.fields})"


class KeyNotFoundError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(f"The key '{invalid_keys}' is not in the database.")


def check_nested_keys(keys_found, data, keys):
    for key in data.keys():
        for i in range(len(keys)):
            if keys[i] in data[key].keys():
                keys_found[i] = True


def check_json_file_for_key(keys_found, file_path, *keys_to_check):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
        return check_nested_keys(keys_found, data, keys_to_check)


def test_multiple_json_files(files_to_test, *keys_to_find):
    keys_found = np.zeros(len(keys_to_find)).astype(bool)

    for file_path in files_to_test:
        check_json_file_for_key(keys_found, file_path, *keys_to_find)

    if np.sum(keys_found) != len(keys_found):
        raise KeyNotFoundError(np.array(keys_to_find)[~keys_found])
