from __future__ import annotations

import maria
from maria import Planner
from maria.io import fetch


def test_pattern_speed():
    input_map = maria.map.load(fetch("maps/crab_nebula.fits"), nu=93e9)

    planner = Planner(target=input_map, site="green_bank", el_bounds=(60, 90))
    plan = planner.generate_plan(
        total_duration=900, scan_pattern="daisy", scan_options={"radius": input_map.width.deg / 3}, sample_rate=50
    )

    plan.plot()
    print(plan)
