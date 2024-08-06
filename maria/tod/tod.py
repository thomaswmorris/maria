import functools
import json
import warnings

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from astropy.io import fits

from maria import utils

from ..coords import Coordinates


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
            components=self.components,
            units=self.units,
            dets=self.dets,
            abscal=self.abscal,
            dtype=self.dtype,
        )

    def __init__(
        self,
        coords: Coordinates,
        components: dict = {},
        units: dict = {},
        dets: pd.DataFrame = None,
        abscal: float = 1.0,
        dtype=np.float32,
    ):
        self.coords = coords
        self.dets = dets
        self.header = fits.header.Header()
        self.abscal = abscal
        self.units = units
        self.dtype = dtype

        self.components = {}

        for field, data in components.items():
            self.components[field] = (
                data if isinstance(data, da.Array) else da.from_array(data)
            )
            self.components[field] = self.components[field].astype(dtype)

        # sort them alphabetically
        self.components = {k: self.components[k] for k in sorted(list(self.fields))}

    def __getattr__(self, attr):
        if attr in self.fields:
            return self.components[attr]
        raise AttributeError(f"No attribute named '{attr}'.")

    def __repr__(self):
        return f"TOD(shape={self.shape}, fields={self.fields})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def fields(self):
        return sorted(list(self.components.keys()))

    @functools.cached_property
    def data(self):
        return sum(self.components.values())

    @functools.cached_property
    def boresight(self):
        return self.coords.boresight

    @property
    def dt(self) -> float:
        return float(np.gradient(self.time, axis=-1).mean())

    @property
    def fs(self) -> float:
        return float(1 / self.dt)

    @property
    def nd(self) -> int:
        return int(self.data.shape[0])

    @property
    def nt(self) -> int:
        return int(self.data.shape[-1])

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
                location=self.location,
                frame="az_el",
            )

            return TOD(
                components={
                    field: data[det_mask]
                    for field, data in self.components.items()
                    if field in fields
                },
                coords=subset_coords,
                dets=self.dets,
                units=self.units,
            )

        if det_mask is not None:
            if not (len(det_mask) == self.nd):
                raise ValueError("The detector mask must have shape (n_dets,).")

            subset_dets = self.dets.loc[det_mask] if self.dets is not None else None

            subset_coords = Coordinates(
                time=self.time,
                phi=self.coords.az[det_mask],
                theta=self.coords.el[det_mask],
                location=self.location,
                frame="az_el",
            )

            return TOD(
                components={
                    field: data[det_mask]
                    for field, data in self.components.items()
                    if field in fields
                },
                coords=subset_coords,
                dets=subset_dets,
                units=self.units,
            )

    def process(self, **kwargs):
        D = self.data.compute()
        W = np.ones(D.shape)

        if "window" in kwargs:
            if "tukey" in kwargs["window"]:
                W *= sp.signal.windows.tukey(
                    D.shape[-1], alpha=kwargs["window"]["tukey"].get("alpha", 0.1)
                )
                D = W * sp.signal.detrend(D, axis=-1)

        if "filter" in kwargs:
            if "window" not in kwargs:
                warnings.warn("Filtering without windowing is not recommended.")

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
                spacing=kwargs["despline"].get("knot_spacing", 10),
                order=kwargs["despline"].get("spline_order", 3),
            )

            A = np.linalg.inv(B @ B.T) @ B @ D.T
            D -= A.T @ B

        return W, D

    @property
    def time(self):
        return self.coords.time

    @property
    def location(self):
        return self.coords.location

    @property
    def lat(self):
        return np.round(self.location.lat.deg, 6)

    @property
    def lon(self):
        return np.round(self.location.lon.deg, 6)

    @property
    def alt(self):
        return np.round(self.location.height.value, 6)

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
                array=self.data.flatten(),
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
            f.createcomponentsset(fname)

    def plot(self, calibrate=True, detrend=True, mean=True, n_freq_bins: int = 256):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4),
            dpi=256,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1, 1.6]},
        )
        ps_ax, tod_ax = axes

        for field, data in self.components.items():
            for band_name in np.unique(self.dets.band_name):
                band_mask = self.dets.band_name == band_name
                d = data[band_mask]

                if detrend:
                    d = sp.signal.detrend(d)
                if calibrate:
                    d *= self.dets.cal.loc[band_mask].mean()

                f, ps = sp.signal.periodogram(
                    sp.signal.detrend(d), fs=self.fs, window="tukey"
                )

                f_bins = np.geomspace(f[1], f[-1], n_freq_bins)
                f_mids = np.sqrt(f_bins[1:] * f_bins[:-1])

                binned_ps = sp.stats.binned_statistic(
                    f, ps.mean(axis=0), bins=f_bins, statistic="mean"
                )[0]

                use = binned_ps > 0

                ps_ax.plot(f_mids[use], binned_ps[use], label=f"{band_name} {field}")
                tod_ax.plot(
                    self.time - self.time.min(),
                    d[0],
                    lw=5e-1,
                    label=f"{band_name} {field}",
                )

        ps_ax.set_xlabel("Time [s]")
        ps_ax.set_ylabel("$T_{RJ}$ [K]" if calibrate else "[pW]")
        ps_ax.legend()
        ps_ax.loglog()
        ps_ax.set_xlim(f_mids.min(), f_mids.max())

        ps_ax.set_xlabel("Frequency [Hz]")

        ylabel = (
            r"Power spectrum [$K_{\rm RJ}^2~{\rm Hz}^{-1}$]"
            if calibrate
            else r"Power spectrum [pW$^2$Hz$^{-1}$]"
        )

        tod_ax.set_xlim(0, float(self.time.max()))
        tod_ax.set_ylabel(ylabel)


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
