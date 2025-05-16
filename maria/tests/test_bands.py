from __future__ import annotations

import numpy as np
import pytest

from maria.band import Band, BandList, all_bands, get_band, parse_band
from maria.errors import FrequencyOutOfBoundsError


def test_band_conventions():
    band1 = Band(center=150e9, width=30e9, NET_RJ=1e-5)
    band2 = {"center": 90e9, "width": 30e9, "NEP": 1e-15}
    band3 = "act/pa5/f150"

    for band in [band1, band2, band3]:
        parse_band(band)

    bl1 = BandList(bands=[band1, band2, band3])

    assert bl1[1].name == "f090"

    bl2 = BandList(bands={"first_band": band1, "second_band": band2, "third_band": band3})

    assert bl2[1].name == "second_band"


def test_frequency_limits():
    caught = False
    try:
        Band(center=90e3, width=20e3)
    except FrequencyOutOfBoundsError:
        caught = True
    assert caught

    caught = False
    try:
        Band(center=90e15, width=20e15)
        caught = False
    except FrequencyOutOfBoundsError:
        caught = True
    assert caught


def test_band_manual():
    nu = np.linspace(120e9, 180e9, 64)
    tau = np.exp(-(((nu - 150e9) / (2 * 10e9)) ** 2))
    b = Band(nu=nu, tau=tau)
    b.plot()


def test_band_list():
    bl = BandList()
    for band_name in all_bands:
        band = get_band(band_name)
        bl.add(band)


def test_noise_conversion():
    my_band = Band(
        center=150e9,  # in Hz
        width=30e9,  # in Hz
        efficiency=0.5,  # in K_RJ
        NET_RJ=1e-5,
        spectrum_kwargs={
            "region": "chajnantor",
            "pwv": 1e1,  # in mm
            "elevation": 90,
        },
    )  # in degrees
    print(my_band)
