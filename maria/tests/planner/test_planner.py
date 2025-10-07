from __future__ import annotations

import maria
from maria import Planner
from maria.errors import PointingError
from maria.io import fetch


def test_planner():
    input_map = maria.map.load(fetch("maps/crab_nebula.fits"), nu=93e9)

    planner = Planner(
        target=input_map, site="green_bank", constraints={"el": (70, 90), "min_sun_distance": 20, "hour": (14, 15)}
    )

    plans = planner.generate_plans(
        total_duration=900, scan_pattern="daisy", scan_options={"radius": input_map.width.deg / 3}, sample_rate=50
    )

    plans[0].plot()
    print(plans)


def test_planner_error():
    input_map = maria.map.load(fetch("maps/crab_nebula.fits"), nu=93e9)

    planner = Planner(target=input_map, site="amundsen_scott", constraints={"el": (70, 90)})

    try:
        planner.generate_plans(
            total_duration=900,
        )
    except PointingError:
        pass

    else:
        assert False
