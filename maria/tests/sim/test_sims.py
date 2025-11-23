from __future__ import annotations

import os

import maria
import numpy as np
import pytest
from maria import Simulation, all_instruments, all_sites, get_plan
from maria.mappers import BinMapper
from maria.utils import read_yaml

here, this_filename = os.path.split(__file__)

# all_test_plan_configs = list(read_yaml(f"{here}/../configs/test_plans.yml").values())

all_instruments.pop(all_instruments.index("alma/ALMA"))

n_sims = 10
test_instruments = np.random.choice(a=all_instruments, size=n_sims)
test_sites = np.random.choice(a=all_sites, size=n_sims)
test_az = np.random.uniform(low=0, high=360, size=n_sims)
test_el = np.random.uniform(low=30, high=90, size=n_sims)


@pytest.mark.parametrize(
    "instrument,site,az,el",
    zip(test_instruments, test_sites, test_az, test_el),
)
def test_pipeline(instrument, site, az, el):
    site = maria.get_site(site)

    plan = maria.Plan.generate(
        scan_pattern="daisy",
        scan_options={"radius": 0.5, "speed": 0.1},  # in degrees
        duration=10,  # in seconds
        sample_rate=15,  # in Hz
        scan_center=(az, el),
        frame="az/el",
    )

    sim = Simulation(
        instrument=instrument, site=site, plans=[plan], atmosphere="2d", cmb="generate", cmb_kwargs={"nside": 256}
    )

    tod = sim.run()[0]

    for field in ["atmosphere", "cmb"]:
        if np.isnan(tod.data[field]).any():
            raise ValueError(f"There are NaNs in the '{field}' field.")

    tod = tod.to("K_RJ")

    mapper = BinMapper(
        tod_preprocessing={
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
            "remove_modes": {"modes_to_remove": 1},
        },
        map_postprocessing={
            "gaussian_filter": {"sigma": 1},
        },
        units="mK_RJ",
        tods=[tod],
    )

    output_map = mapper.run()

    assert output_map.weight.sum() > 0
