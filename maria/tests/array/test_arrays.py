from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest

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

arrays2 = [{"file": "data/alma/alma_f144.csv", "primary_size": 12, "bands": ["alma/f144"]}]


offsets = np.random.uniform(low=0, high=1, size=(100, 2))
arrays3 = [{"bands": ["act/pa4/f220"], "primary_size": 5, "xi": offsets[..., 0], "eta": offsets[..., 1], "degrees": True}]

array_configs_to_test = [arrays1, arrays2, arrays3]


@pytest.mark.parametrize("array_config", array_configs_to_test)
def test_generate_array(array_config):
    instrument = maria.Instrument(arrays=array_config)


def test_array_from_text_file(tmp_path):
    table = np.c_[np.random.uniform(-600, 600, size=(256, 2)), np.random.uniform(50, 500, size=256)]
    np.savetxt(tmp_path / "dets.dat", table, fmt="%8.1f")
    xi, eta, nefd = np.loadtxt(tmp_path / "dets.dat").T

    array = maria.Array.from_kwargs(
        file=tmp_path / "dets.dat",
        columns=["xi", "eta", "NEFD"],
        units={"xi": "arcsec", "eta": "arcsec", "NEFD": "mJy√s"},
        bands=["apex/350um"],
        primary_size=12,
    )

    assert np.allclose(np.degrees(array.xi) * 3600, xi, rtol=1e-9)
    assert np.allclose(np.degrees(array.eta) * 3600, eta, rtol=1e-9)

    # converting the NEP back with the same beam should give the NEFD
    band = array.bands[0]
    fwhm = array.fwhm.rad
    beam_area = np.pi * fwhm**2 / (4 * np.log(2))
    array_nefd = band.cal("W -> Jy/beam", nu=band.center.Hz, beam_area=beam_area)(array.NEP)
    assert np.allclose(1e3 * array_nefd, nefd, rtol=1e-9)


def test_per_detector_sensitivity():
    band = maria.Band(name="f150", center=150e9, width=30e9, NET_RJ=5e-5)
    offsets = np.random.uniform(low=0, high=1, size=(100, 2))

    array = maria.Array.from_kwargs(
        bands=[band], primary_size=5, xi=offsets[:, 0], eta=offsets[:, 1], NET_RJ=np.full(100, 5e-5)
    )
    assert np.allclose(array.NEP, band.NEP.to("W√s"), rtol=1e-12)
    assert np.all(array.noise_weight() == 1)

    array = maria.Array.from_kwargs(bands=[band], primary_size=5, xi=offsets[:, 0], eta=offsets[:, 1])
    assert np.isnan(array.dets.NEP).all()
    assert np.all(array.NEP == band.NEP.to("W√s"))

    # a per-detector sensitivity is ambiguous if every detector is in every band
    with pytest.raises(ValueError):
        maria.Array.from_kwargs(
            bands=["act/pa5/f090", "act/pa5/f150"],
            primary_size=5,
            xi=offsets[:, 0],
            eta=offsets[:, 1],
            NET_RJ=np.full(100, 5e-5),
        )


def test_band_without_noise_level():
    band = maria.Band(name="f150", center=150e9, width=30e9, NEP=np.nan)
    offsets = np.random.uniform(low=0, high=1, size=(100, 2))
    array = {"bands": [band], "primary_size": 5, "xi": offsets[:, 0], "eta": offsets[:, 1]}

    # the detectors supply the noise level, so they all need one
    assert np.all(maria.Array.from_kwargs(**array, NEP=np.full(100, 1e-17)).NEP == 1e-17)
    with pytest.raises(ValueError):
        maria.Array.from_kwargs(**array)
