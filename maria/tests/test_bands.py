from __future__ import annotations

import numpy as np
import pytest

from maria.instrument import Band


def test_band_auto():
    b = Band(center=130, width=10)
    b.plot()
    print(b.dP_dTRJ)


def test_band_manual():
    nu = np.linspace(120, 180, 64)
    tau = np.exp(-(((nu - 150) / (2 * 10)) ** 2))

    b = Band(nu=nu, tau=tau, center=100)
    b.plot()
    print(b.dP_dTRJ)
