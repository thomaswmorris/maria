import pytest

import maria

dets1 = {
    "subarray-1": {
        "n": 500,
        "field_of_view": 2,
        "array_shape": "hex",
        "bands": [{"center": 30, "width": 5}, {"center": 40, "width": 5}],
    },
    "subarray-2": {
        "n": 500,
        "field_of_view": 2,
        "array_shape": "hex",
        "bands": [{"center": 90, "width": 5}, {"center": 150, "width": 5}],
    },
}

dets2 = {
    "n": 500,
    "field_of_view": 2,
    "array_shape": "hex",
    "bands": ["alma/f043", "alma/f078"],
}

dets3 = {"file": "data/alma/alma.cycle1.total.csv"}


@pytest.mark.parametrize("instrument_name", maria.all_instruments)
def test_get_instrument(instrument_name):
    instrument = maria.get_instrument(instrument_name)


@pytest.mark.parametrize("dets", [dets1, dets2, dets3])
def test_get_custom_array(dets):
    instrument = maria.get_instrument(dets=dets)
