from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import maria
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
    map_filename = fetch("maps/sun.h5")
    m = maria.map.load(filename=map_filename)
    plan = maria.Plan(start_time=0)
    sim = maria.Simulation(plan=plan, map=m)
    tod = sim.run()
    tod.plot()
