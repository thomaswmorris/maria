from __future__ import annotations

import os

import numpy as np
import pytest

import maria
from maria import Simulation, all_instruments, all_sites, get_plan
from maria.mappers import BinMapper
from maria.utils import read_yaml

here, this_filename = os.path.split(__file__)

all_test_plan_configs = list(read_yaml(f"{here}/configs/test_plans.yml").values())

all_instruments.pop(all_instruments.index("alma/ALMA"))

n_sims = 8
test_instruments = np.random.choice(a=all_instruments, size=n_sims)
test_sites = np.random.choice(a=all_sites, size=n_sims)


@pytest.mark.parametrize(
    "instrument,site",
    zip(test_instruments, test_sites),
)
def test_pipeline(instrument, site):
    site = maria.get_site(site)

    plan = maria.Plan(
        start_time="2024-08-06T09:00:00",
        scan_pattern="daisy",
        scan_options={"radius": 0.5, "speed": 0.1},  # in degrees
        duration=5,  # in seconds
        sample_rate=20,  # in Hz
        scan_center=(31, 62),
        frame="az_el",
    )

    sim = Simulation(
        instrument=instrument,
        site=site,
        plan=plan,
        atmosphere="2d",
        cmb="generate",
    )

    tod = sim.run()

    for field in ["atmosphere", "cmb"]:
        if np.isnan(tod.data[field]).any():
            raise ValueError(f"There are NaNs in the '{field}' field.")

    tod = tod.to("K_RJ")

    mapper = BinMapper(
        center=(0, -23),
        stokes="IQUV",
        frame="ra_dec",
        width=1,
        height=1,
        resolution=1 / 256,
        tod_preprocessing={
            "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
            "remove_modes": {"modes_to_remove": [0]},
        },
        map_postprocessing={
            "gaussian_filter": {"sigma": 1},
            "median_filter": {"size": 1},
        },
        units="mK_RJ",
    )

    mapper.add_tods(tod)
    mapper.run()
