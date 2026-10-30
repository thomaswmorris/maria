from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from maria.instrument import Instrument

plt.close("all")

# subarray_dets = {
#     "subarray-1": {
#         "n": 500,
#         "field_of_view": 2,
#         "array_shape": "hex",
#         "bands": [{"center": 30, "width": 5}, {"center": 40, "width": 5}],
#     },
#     "subarray-2": {
#         "n": 500,
#         "field_of_view": 2,
#         "array_shape": "hex",
#         "bands": [{"center": 90, "width": 5}, {"center": 150, "width": 5}],
#     },
# }

# predefined_bands = {
#     "n": 500,
#     "field_of_view": 2,
#     "array_packing": "square",
#     "array_shape": "hex",
#     "bands": ["alma/f043", "alma/f078"],
# }

# predefined_dets = {"file": "data/alma/alma.cycle1.total.csv"}

# beam_packing_dets = {
#     "subarray-1": {
#         "field_of_view": 0.2,
#         "beam_spacing": 1,
#         "primary_size": 10,
#         "array_shape": "hex",
#         "array_packing": "sunflower",
#         "band": {"center": 30, "width": 5},
#     },
#     "subarray-2": {
#         "n": 500,
#         "field_of_view": 0.2,
#         "array_shape": "circle",
#         "array_packing": "hex",
#         "bands": [{"center": 90, "width": 5}, {"center": 150, "width": 5}],
#     },
# }


@pytest.mark.parametrize("instrument_name", maria.all_instruments)
def test_get_instrument(instrument_name):
    instrument = maria.get_instrument(instrument_name)
    print(instrument)

    instrument.plot()

    plt.close("all")


# @pytest.mark.parametrize(
#     "dets", [subarray_dets, predefined_bands, predefined_dets, beam_packing_dets]
# )
# def test_get_custom_array(dets):
#     instrument = maria.get_instrument(dets=dets)


def assert_fits_round_trip(instrument, path):
    instrument.to_fits(path)
    new_instrument = Instrument.from_fits(path)

    for array, new_array in zip(instrument.arrays, new_instrument.arrays):
        pd.testing.assert_frame_equal(array.dets, new_array.dets)
        for band, new_band in zip(array.bands, new_array.bands):
            assert np.array_equal(band.nu.Hz, new_band.nu.Hz)
            assert np.array_equal(band.tau, new_band.tau)
            for attr in ["name", "efficiency", "knee", "time_constant", "gain_error", "NEP", "NET_RJ", "center", "width"]:
                assert getattr(band, attr) == getattr(new_band, attr)

    assert np.array_equal(instrument.dets.NEP, new_instrument.dets.NEP)


@pytest.mark.parametrize("instrument_name", ["test/1deg", "MUSTANG-2", "act/AdvACT", "alma/ALMA", "so/sat"])
def test_instrument_fits_round_trip(instrument_name, tmp_path):
    assert_fits_round_trip(maria.get_instrument(instrument_name), tmp_path / "instrument.fits")


@pytest.mark.internet
def test_instrument_fits_round_trip_with_spectrum(tmp_path):
    band = maria.Band(
        name="f150",
        center=150e9,
        width=30e9,
        NET_RJ=5e-5,
        spectrum_kwargs={"region": "chajnantor", "pwv": 1, "elevation": 60},
    )
    instrument = maria.get_instrument(array={"n": 100, "field_of_view": 0.1, "primary_size": 5, "bands": [band]})
    assert_fits_round_trip(instrument, tmp_path / "instrument.fits")
