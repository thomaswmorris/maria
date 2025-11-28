from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
from maria.io import fetch
from maria.map import ProjectionMap

plt.close("all")


def test_map_extend():
    map_filename = fetch("maps/cluster1.fits")

    m1 = maria.map.load(filename=map_filename, nu=90e9)
    m2 = maria.map.load(filename=map_filename, nu=150e9)
    m3 = maria.map.load(filename=map_filename, nu=220e9)

    m4 = m1.extend([m2, m3], dim="nu").unsqueeze("stokes")
    m5, m6 = m4.copy(), m4.copy()
    m5.stokes = "Q"
    m6.stokes = "U"

    m4.extend([m5, m6], dim="stokes")


def test_map_slice():
    stokes = "IQUV"
    nu = [90e9, 150e9, 220e9]
    t = 1.7e9 + np.arange(0, 600, 120)
    data = np.random.standard_normal((len(stokes), len(nu), len(t), 100, 100))

    m = ProjectionMap(data=data, width=1e0, stokes=stokes, nu=nu, t=t, center=(0, -30), frame="ra_dec")

    m[0, :, ::2, :2]
