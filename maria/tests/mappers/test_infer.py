from __future__ import annotations

import os

import maria
from maria.instrument import Band
from maria.io import fetch
from maria.mappers import BinMapper

here, this_filename = os.path.split(__file__)


def test_mapper_inference():
    map_filename = fetch("maps/cluster1.fits", refresh=True)

    f090 = Band(center=90e9, width=20e9, NET_RJ=5e-5)
    f150 = Band(center=150e9, width=30e9, NET_RJ=5e-5)

    array = {"field_of_view": 0.02, "bands": [f090, f150], "primary_size": 50}

    instrument = maria.get_instrument(array=array)

    map_filename = fetch("maps/cluster1.fits")

    input_map = maria.map.load(filename=map_filename, nu=150e9, width=0.1, center=(150, 10))

    planner = maria.Planner(target=input_map, site="cerro_toco", constraints={"el": (45, 90)})

    plans = planner.generate_plans(
        total_duration=60, sample_rate=50, scan_pattern="daisy", scan_options={"radius": input_map.width.deg / 3}
    )

    sim = maria.Simulation(
        instrument,
        plans=plans,
        site="llano_de_chajnantor",
        map=input_map,
    )

    tods = sim.run()

    BinMapper(tods=tods).run()
    BinMapper(center=(-45, 45), tods=tods).run()
    BinMapper(width=0.45, tods=tods).run()
    BinMapper(resolution=input_map.width.deg / 10, tods=tods).run()
