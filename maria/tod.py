import copy
import json

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits as pyfits

from . import utils
from .coordinator import Coordinator


class TOD:
    """ """

    def __init__(self):
        pass

    def subset(self, mask):
        tod_subset = copy.deepcopy(self)

        tod_subset.data = tod_subset.data[mask]
        tod_subset.detectors = tod_subset.detectors.loc[mask]

        return tod_subset

    def to_fits(self, filename, array="MUSTANG-2"):
        """
        safe tod to fits
        """

        if array == "MUSTANG-2":
            coordinator = Coordinator(
                lat=self.meta["latitude"], lon=self.meta["longitude"]
            )

            self.AZ, self.EL = utils.coords.xy_to_lonlat(
                self.dets.offset_x.values[:, None],
                self.dets.offset_y.values[:, None],
                self.az,
                self.el,
            )

            self.RA, self.DEC = coordinator.transform(
                self.time,
                self.AZ,
                self.EL,
                in_frame="az_el",
                out_frame="ra_dec",
            )

            header = pyfits.header.Header()

            header["AZIMUTH"] = (self.center_az_el[0], "radians")
            header["ELEVATIO"] = (self.center_az_el[1], "radians")
            header["BMAJ"] = (8.0, "arcsec")
            header["BMIN"] = (8.0, "arcsec")
            header["BPA"] = (0.0, "degrees")

            header["SITELAT"] = (self.meta["latitude"], "Site Latitude")
            header["SITELONG"] = (self.meta["longitude"], "Site Longitude")
            header["SITEELEV"] = (self.meta["altitude"], "Site elevation (meters)")

            col01 = pyfits.Column(
                name="DX   ", format="E", array=self.RA.flatten(), unit="radians"
            )
            col02 = pyfits.Column(
                name="DY   ", format="E", array=self.DEC.flatten(), unit="radians"
            )
            col03 = pyfits.Column(
                name="FNU  ", format="E", array=self.data.flatten(), unit="Kelvin"
            )
            col04 = pyfits.Column(name="UFNU ", format="E")
            col05 = pyfits.Column(
                name="TIME ",
                format="E",
                array=((self.time - self.time[0]) * np.ones_like(self.RA)).flatten(),
                unit="s",
            )
            col06 = pyfits.Column(name="COL  ", format="I")
            col07 = pyfits.Column(name="ROW  ", format="I")
            col08 = pyfits.Column(
                name="PIXID",
                format="I",
                array=(
                    np.arange(len(self.RA), dtype=np.int16).reshape(-1, 1)
                    * np.ones_like(self.RA)
                ).flatten(),
            )
            col09 = pyfits.Column(
                name="SCAN ", format="I", array=np.zeros_like(self.RA).flatten()
            )
            col10 = pyfits.Column(name="ELEV ", format="E")

            hdu = pyfits.BinTableHDU.from_columns(
                [col01, col02, col03, col04, col05, col06, col07, col08, col09, col10],
                header=header,
            )

            hdu.writeto(filename, overwrite=True)

    def from_fits(self, filename, array="MUSTANG-2", hdu=1):
        """
        read tod from fits
        """
        if array == "MUSTANG-2":
            f = pyfits.open(filename)
            raw = f[hdu].data

            pixid = raw["PIXID"]
            dets = np.unique(pixid)
            ndet = len(dets)
            nsamp = len(pixid) // len(dets)

            self.header = pyfits.header.Header()
            self.unit = "K"
            self.pntunit = "radians"

            self.RA = np.reshape(raw["DX"], [ndet, nsamp])
            self.DEC = np.reshape(raw["DY"], [ndet, nsamp])
            self.ra = np.mean(self.RA, axis=0)
            self.dec = np.mean(self.DEC, axis=0)

            self.time = np.reshape(raw["TIME"], [ndet, nsamp])
            self.time = np.mean(self.time, axis=0)
            self.cntr = (self.ra.mean(), self.dec.mean())

            self.dets = pd.DataFrame(
                {
                    "band": ["93GHz"] * ndet,
                    "band_center": np.ones(ndet) * 93,
                    "band_width": np.ones(ndet) * 10,
                }
            )

            self.data = np.reshape(raw["FNU"], [ndet, nsamp])

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("")

    def plot(self):
        pass

    @property
    def center_ra_dec(self):
        return utils.coords.get_center_lonlat(self.ra, self.dec)

    @property
    def center_az_el(self):
        return utils.coords.get_center_lonlat(self.az, self.el)


class KeyNotFoundError(Exception):
    def __init__(self, invalid_keys):
        super().__init__(f"The key '{invalid_keys}' is not in the database. ")


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
