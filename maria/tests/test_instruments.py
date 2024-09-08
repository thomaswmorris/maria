import pytest

import maria

subarray_dets = {
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

predefined_bands = {
    "n": 500,
    "field_of_view": 2,
    "array_packing": "square",
    "array_shape": "hex",
    "bands": ["alma/f043", "alma/f078"],
}

predefined_dets = {"file": "data/alma/alma.cycle1.total.csv"}

beam_packing_dets = {
    "subarray-1": {
        "field_of_view": 0.2,
        "beam_spacing": 1,
        "primary_size": 10,
        "array_shape": "hex",
        "array_packing": "sunflower",
        "band": {"center": 30, "width": 5},
    },
    "subarray-2": {
        "n": 500,
        "field_of_view": 0.2,
        "array_shape": "circle",
        "array_packing": "hex",
        "bands": [{"center": 90, "width": 5}, {"center": 150, "width": 5}],
    },
}


@pytest.mark.parametrize("instrument_name", maria.all_instruments)
def test_get_instrument(instrument_name):
    instrument = maria.get_instrument(instrument_name)
    print(instrument)


# @pytest.mark.parametrize(
#     "dets", [subarray_dets, predefined_bands, predefined_dets, beam_packing_dets]
# )
# def test_get_custom_array(dets):
#     instrument = maria.get_instrument(dets=dets)
