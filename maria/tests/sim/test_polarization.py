from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
from maria import Planner, Simulation, all_instruments
from maria.io import fetch
from maria.mappers import BinMapper
from maria.utils import read_yaml


def test_polarized_map_sim():
    einstein = maria.map.load(fetch("maps/einstein.h5"))
    planner = Planner(target=einstein, site="llano_de_chajnantor", constraints={"el": (60, 90)})
    plans = planner.generate_plans(total_duration=10, sample_rate=50)  # in Hz

    plans[0].plot()
    print(plans)

    sim = Simulation(
        instrument="test/1deg",
        site="llano_de_chajnantor",
        plans=plans,
        atmosphere="2d",
        cmb="generate",
        map=einstein,
    )

    tod = sim.run()[0]

    for field in ["atmosphere", "cmb"]:
        if np.isnan(tod.data[field]).any():
            raise ValueError(f"There are NaNs in the '{field}' field.")

    tod = tod.to("K_RJ")

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
