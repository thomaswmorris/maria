from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import maria

plt.close("all")


array1 = {
    "subarray-1": {
        "n": 500,
        "primary_size": 10,
        "field_of_view": 2,
        "shape": "hexagon",
        "bands": [{"center": 30e9, "width": 5e9}, {"center": 40e9, "width": 5e9}],
    },
    "subarray-2": {
        "n": 500,
        "primary_size": 10,
        "field_of_view": 2,
        "shape": "hexagon",
        "bands": [{"center": 90e9, "width": 5e9}, {"center": 150e9, "width": 5e9}],
        "rotation": 90,
    },
}

# /Users/tom/maria/src/maria/array/

array2 = [{"file": "alma/alma_f144.csv"}]

array_configs_to_test = [array1, array2]


@pytest.mark.parametrize("array_config", array_configs_to_test)
def test_generate_array(array_config):
    instrument = maria.Instrument(arrays=array_config)
