import copy
import json

import h5py
import numpy as np
from . import utils


class TOD:
    """ """

    def __init__(self):
        pass

    def subset(self, mask):
        tod_subset = copy.deepcopy(self)

        tod_subset.data = tod_subset.data[mask]
        tod_subset.detectors = tod_subset.detectors.loc[mask]

        return tod_subset

    def to_fits(self, filename):
        """ """

        ...

    def from_fits(self, filename):
        """ """

        ...

    def to_hdf(self, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("")

    def plot(self):
        pass

    @property
    def center_ra_dec(self):
       return utils.get_center_lonlat(self.ra, self.dec)

    @property
    def center_az_el(self):
       return utils.get_center_lonlat(self.az, self.el)

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
