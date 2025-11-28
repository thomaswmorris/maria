from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
from maria import Planner, Simulation, all_instruments
from maria.io import fetch
from maria.map import ProjectionMap
from maria.mappers import BinMapper
from maria.utils import read_yaml


def test_polarized_map_sim():
    nu = [90e9, 150e9, 220e9]
    data = np.random.standard_normal((len(nu), 100, 100))

    multifrequency_map = ProjectionMap(data=data, width=1e0, nu=nu, center=(0, -30), frame="ra_dec")

    planner = Planner(target=multifrequency_map, site="llano_de_chajnantor", constraints={"el": (60, 90)})
    plans = planner.generate_plans(total_duration=10, sample_rate=50)  # in Hz

    plans[0].plot()
    print(plans)

    sim = Simulation(
        instrument="test/1deg",
        site="llano_de_chajnantor",
        plans=plans,
        atmosphere="2d",
        map=multifrequency_map,
    )

    tod = sim.run()[0]

    mapper = BinMapper(
        center=(0, -23),
        stokes="IQUV",
        frame="ra/dec",
        width=1,
        height=1,
        resolution=1 / 256,
        tod_preprocessing={
            "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
            "remove_modes": {"modes_to_remove": 1},
        },
        map_postprocessing={
            "gaussian_filter": {"sigma": 1},
            "median_filter": {"size": 1},
        },
        units="mK_RJ",
        tods=[tod],
    )

    output_map = mapper.run()

    output_map.to("Jy/pixel").plot()

    plt.close("all")
