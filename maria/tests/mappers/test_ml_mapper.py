from __future__ import annotations

import maria
import matplotlib.pyplot as plt
from maria.instrument import Band
from maria.io import fetch
from maria.mappers import MaximumLikelihoodMapper


def test_ml_mapper():
    map_filename = fetch("maps/cluster1.fits", refresh=True)

    f090 = Band(center=90e9, width=20e9, NET_RJ=5e-5)
    f150 = Band(center=150e9, width=30e9, NET_RJ=5e-5)

    array = {"field_of_view": 0.02, "bands": [f090, f150], "primary_size": 50}

    instrument = maria.get_instrument(array=array)

    map_filename = fetch("maps/cluster1.fits")

    input_map = maria.map.load(filename=map_filename, nu=150e9, width=0.1, center=(150, 10))

    input_map.data *= 1e3

    planner = maria.Planner(target=input_map, site="cerro_toco", constraints={"el": (45, 90)})

    plans = planner.generate_plans(
        total_duration=60, sample_rate=50, scan_pattern="daisy", scan_options={"radius": input_map.width.deg / 3}
    )

    sim = maria.Simulation(
        instrument,
        plans=plans,
        site="llano_de_chajnantor",
        map=input_map,
        atmosphere="2d",
    )

    tods = sim.run()

    mapper = MaximumLikelihoodMapper(
        tods=tods,
    )

    mapper.fit()
    mapper.map.to("Jy/beam").plot()

    plt.close("all")
