import json
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import scipy as sp
from astropy.io import fits

from .. import site
from ..coords import Coordinates, get_center_phi_theta
from ..instrument.dets import Detectors


class TOD:
    """
    Time-ordered data.
    """

    @property
    @lru_cache(maxsize=None)
    def data(self):
        """
        Combine all the fields into the total.
        """
        return sum(self._data.values())  # * self.abscal

    @property
    @lru_cache(maxsize=None)
    def data_calibrated(self):
        """
        Combine all the fields into the total.
        """
        T_ant = np.zeros(self.data.shape)
        if "atmosphere" in self._data.keys():
            T_ant += 0.8 * self._data["atmosphere"]
        if "map" in self._data.keys():
            T_ant += 0.3 * self._data["map"]
        if "noise" in self._data.keys():
            T_ant += 1.0 * self._data["noise"]
        return T_ant / 0.3 * self.abscal

    @staticmethod
    def from_fits(fname: str, format: str, **kwargs):
        if format.lower() == "mustang-2":
            return TOD._from_mustang2(fname=fname, **kwargs)

        if format.lower() == "atlast":
            return TOD._from_atlast(fname=fname, **kwargs)

        if format.lower() == "abs":
            ...

        if format.lower() == "act":
            ...

    @classmethod
    def _from_atlast(
        cls, fname: str, hdu: int = 1, band_center: int = 93, band_width: int = 52
    ):
        f = fits.open(fname)
        raw = f[hdu].data

        det_uids, det_counts = np.unique(raw["PIXID"], return_counts=True)

        if det_counts.std() > 0:
            raise ValueError("Cannot reshape a ragged TOD.")

        n_dets = len(det_uids)
        n_samp = det_counts.max()

        data = {"data": raw["FNU"].astype("float32").reshape((n_dets, n_samp))}

        ra = raw["dx"].astype(float).reshape((n_dets, n_samp))
        dec = raw["dy"].astype(float).reshape((n_dets, n_samp))
        t = 1.6e9 + raw["time"].astype(float).reshape((n_dets, n_samp)).mean(axis=0)

        coords = Coordinates(
            time=t,
            phi=ra,
            theta=dec,
            location=site.get_location("llano_de_chajnantor"),
            frame="ra_dec",
        )

        dets = Detectors.generate(
            bands_config={
                "f093": {
                    "n_dets": n_dets,
                    "band_center": band_center,
                    "band_width": band_width,
                }
            }
        )

        return cls(coords=coords, dets=dets, data=data, units={"data": "K"})

    @classmethod
    def _from_mustang2(cls, fname: str, hdu: int = 1):
        f = fits.open(fname)
        raw = f[hdu].data

        det_uids, det_counts = np.unique(raw["PIXID"], return_counts=True)

        if det_counts.std() > 0:
            raise ValueError("Cannot reshape a ragged TOD.")

        n_dets = len(det_uids)
        n_samp = det_counts.max()

        data = {"data": raw["FNU"].astype("float32").reshape((n_dets, n_samp))}

        ra = raw["dx"].astype(float).reshape((n_dets, n_samp))
        dec = raw["dy"].astype(float).reshape((n_dets, n_samp))
        t = 1.6e9 + raw["time"].astype(float).reshape((n_dets, n_samp)).mean(axis=0)

        coords = Coordinates(
            time=t,
            phi=ra,
            theta=dec,
            location=site.get_location("green_bank"),
            frame="ra_dec",
        )

        dets = Detectors.generate(
            bands_config={
                "f093": {
                    "n_dets": n_dets,
                    "band_center": 93,
                    "band_width": 30,
                }
            }
        )

        return cls(coords=coords, dets=dets, data=data, units={"data": "K"})

    def __init__(
        self,
        coords: Coordinates,
        data: dict = {},
        units: dict = {},
        dets: pd.DataFrame = None,
        boresight: Coordinates = None,
        abscal: float = 1.0,
    ):
        self.coords = coords
        self.dets = dets
        self._data = data
        self.header = fits.header.Header()
        self.abscal = abscal

        # if no boresight is supplied, infer it from the inputs
        if boresight is not None:
            self.boresight = boresight
        else:
            boresight_az, boresight_el = get_center_phi_theta(
                coords.az, coords.el, keep_last_dim=True
            )
            self.boresight = Coordinates(
                time=coords.time,
                phi=boresight_az,
                theta=boresight_el,
                location=coords.location,
                frame="az_el",
            )

        self.units = units

    @property
    def dt(self):
        return np.diff(self.time).mean()

    @property
    def fs(self):
        return 1 / self.dt

    @property
    def nd(self):
        return self.data.shape[0]

    @property
    def nt(self):
        return self.data.shape[-1]

    def subset(self, det_mask=None, time_mask=None, band: str = None):
        if band is not None:
            det_mask = self.dets.band == band
            if not det_mask.sum() > 0:
                raise ValueError(f"There are no detectors for band '{band}'.")
            return self.subset(det_mask=det_mask)

        if time_mask is not None:
            if not (len(time_mask) == self.nt):
                raise ValueError("The detector mask must have shape (n_dets,).")

            subset_coords = Coordinates(
                time=self.time[time_mask],
                phi=self.coords.az[:, time_mask],
                theta=self.coords.el[:, time_mask],
                location=self.location,
                frame="az_el",
            )

            subset_data = {}
            for k, v in self._data.items():
                subset_data[k] = v[:, time_mask]

            return TOD(
                data=subset_data,
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

            subset_data = {}
            for k, v in self._data.items():
                subset_data[k] = v[det_mask]

            return TOD(
                data=subset_data,
                coords=subset_coords,
                dets=subset_dets,
                units=self.units,
            )

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
                array=self._data["data"].flatten(),
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
            f.create_dataset(fname)


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
