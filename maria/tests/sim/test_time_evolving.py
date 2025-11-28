from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.instrument import Band
from maria.io import fetch
from maria.map import all_maps
from maria.mappers import BinMapper

plt.close("all")


def test_time_ordered_map_sim():
    time_evolving_sun_path = fetch("maps/sun.h5")
    input_map = maria.map.load(filename=time_evolving_sun_path, nu=100e9, t=1.7e9 + np.linspace(0, 180, 16))
    plan = maria.Plan.generate(
        start_time=1.7e9,
        duration=180,
        scan_center=(input_map.center),
        scan_options={"radius": 0.25},
    )
    sim = maria.Simulation(instrument="test/1deg", site="cerro_toco", plans=plan, map=input_map)
    tods = sim.run()

    mapper = BinMapper(tods=tods, timestep=60)
    mapper.run()

    plt.close("all")
