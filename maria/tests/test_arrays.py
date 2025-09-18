from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import maria

plt.close("all")


arrays1 = {
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

arrays2 = [{"file": "alma/alma_f144.csv", "primary_size": 12, "bands": ["alma/f144"]}]

arrays3 = [{"file": "alma/alma_f144.csv", "primary_size": 12, "bands": ["alma/f144"]}]


array_configs_to_test = [arrays1, arrays2, arrays3]


@pytest.mark.parametrize("array_config", array_configs_to_test)
def test_generate_array(array_config):
    instrument = maria.Instrument(arrays=array_config)
