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


def test_recover_map():
    f090 = Band(center=90e9, width=30e9)
    f150 = Band(center=150e9, width=40e9)
    f220 = Band(center=220e9, width=50e9)

    map_filename = fetch("maps/cluster1.fits")
    input_map = maria.map.load(filename=map_filename, nu=150e9).to("K_RJ")[..., ::8, ::8]

    input_map.data -= input_map.data.mean().compute()
    map_width = input_map.width.degrees

    array = {
        "field_of_view": map_width / 2,
        "primary_size": 1000,
        "n": 300,
        "bands": [f090, f150, f220],
    }

    instrument = maria.get_instrument(array=array)

    planner = maria.Planner(target=input_map, site="cerro_toco", constraints={"el": (40, 90)})

    plans = planner.generate_plans(
        total_duration=60, sample_rate=50, scan_pattern="daisy", scan_options={"radius": input_map.width.deg / 3}
    )

    sim = maria.Simulation(
        instrument,
        plans=plans,
        map=input_map,
        site="cerro_toco",
        noise=False,
    )

    tods = sim.run()

    mapper = BinMapper(
        center=input_map.center,
        frame=input_map.frame,
        width=input_map.width,
        resolution=input_map.x_res,
        degrees=False,
        tods=tods,
    )

    mapper.add_tods(tods)
    output_map = mapper.run()

    m0 = input_map.data[0, 0, 0].compute()
    m1 = output_map.data[0, :].compute()
    w = output_map.weight[-1, 0].compute()

    relsqres = np.sqrt(np.nansum(w * (m1 - m0) ** 2, axis=(-1, -2)) / np.nansum(w))

    assert all(relsqres < 1e-3)
