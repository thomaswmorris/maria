from __future__ import annotations

import numpy as np
import pytest

from maria.band import Band, BandList, all_bands, get_band


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
