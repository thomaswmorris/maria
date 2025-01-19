from __future__ import annotations

import numpy as np
import pytest

from maria.band import Band, BandList, all_bands, get_band


def test_band_manual():
    nu = np.linspace(120, 180, 64)
    tau = np.exp(-(((nu - 150) / (2 * 10)) ** 2))
    b = Band(nu=nu, tau=tau)
    b.plot()


@pytest.mark.parametrize("band_name", all_bands)
def test_band_list(band_name):

    bl = BandList()
    for band_name in all_bands:
        band = get_band(band_name)
        bl.add(band)
