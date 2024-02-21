import pytest

import maria

f090 = {"center": 90, "width": 10}
f150 = {"center": 150, "width": 20}

dets1 = {
    "array1": {
        "n": 500,
        "field_of_view": 2,
        "array_shape": "hex",
        "bands": [f090, f150],
    },
    "array2": {
        "n": 500,
        "field_of_view": 2,
        "array_shape": "hex",
        "bands": [f090, f150],
    },
}

dets2 = {
    "n": 500,
    "field_of_view": 2,
    "array_shape": "hex",
    "bands": {"f090": {"center": 90, "width": 10}},
}


@pytest.mark.parametrize("instrument_name", maria.all_instruments)
def test_get_instrument(instrument_name):
    instrument = maria.get_instrument(instrument_name)


@pytest.mark.parametrize("dets", [dets1, dets2])
def test_get_custom_array(dets):
    instrument = maria.get_instrument(dets=dets)
