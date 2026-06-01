from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.instrument import Band
from maria.io import fetch
from maria.map import all_maps
from maria.mappers import BinMapper
from maria.plan import Planner

plt.close("all")


def test_time_ordered_map_sim():

    input_map = maria.map.get(
        "maps/time_evolving_sun.fits",
        nu=100e9,
        t=1.8e9 + np.linspace(0, 180, 16),
        frame="az/el",
        center=(45, 45),
    )

    plans = Planner(target=input_map, site="cerro_chajnantor").generate_plans(
        total_duration=60,
        scan_options={"radius": 0.25},
    )

    sim = maria.Simulation(instrument="test/1deg", site="cerro_toco", plans=plans, map=input_map)
    tods = sim.run()

    mapper = BinMapper(tods=tods, timestep=60)
    mapper.run()

    plt.close("all")
