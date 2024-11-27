from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import maria
import numpy as np
from maria.io import fetch

plt.close("all")


@pytest.mark.parametrize(
    "map_name",
    ["maps/cluster.fits", "maps/big_cluster.fits", "maps/galaxy.fits"],
)  # noqa
def test_fetch_fits_map(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(
        filename=map_filename,
        nu=90,
        resolution=1 / 1024,
        center=(150, 10),
        units="Jy/pixel",
    )

    m.plot()


@pytest.mark.parametrize("map_name", ["maps/sun.h5"])  # noqa
def test_fetch_hdf_map(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(filename=map_filename)
    m.plot()


def test_time_ordered_map_sim():

    time_evolving_sun_path = fetch("maps/sun.h5")
    input_map = maria.map.load(
        filename=time_evolving_sun_path, t=1.7e9 + np.linspace(0, 180, 16)
    )
    plan = maria.Plan(
        start_time=1.7e9,
        duration=180,
        scan_center=np.degrees(input_map.center),
        scan_options={"radius": 0.25},
    )
    sim = maria.Simulation(plan=plan, map=input_map)
    tod = sim.run()
    tod.to("K_RJ").plot()
