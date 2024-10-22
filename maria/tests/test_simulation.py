import os

import numpy as np
import pytest

from maria import Simulation, all_instruments, all_plans, all_sites
from maria.io import read_yaml
from maria.plan import Plan

here, this_filename = os.path.split(__file__)

all_test_plan_configs = list(read_yaml(f"{here}/configs/test_plans.yml").values())

all_instruments.pop(all_instruments.index("alma/ALMA"))

n_sims = 8
test_instruments = np.random.choice(a=all_instruments, size=n_sims)
test_sites = np.random.choice(a=all_sites, size=n_sims)
test_plan_configs = np.random.choice(a=all_test_plan_configs, size=n_sims)


@pytest.mark.parametrize(
    "instrument,site,plan_config", zip(test_instruments, test_sites, test_plan_configs)
)
def test_complete_sim(instrument, site, plan_config):
    plan = Plan(**plan_config)

    sim = Simulation(
        instrument=instrument, site=site, plan=plan, atmosphere="2d", cmb="generate"
    )

    tod = sim.run()

    for field in ["atmosphere", "cmb"]:
        if np.isnan(tod.get_field(field)).any():
            raise ValueError(f"There are NaNs in the '{field}' field.")

    tod = tod.to("K_RJ")

    tod.plot()

    tod.process(despline={"knot_spacing": 60}).twinkle(
        rate=2, max_frames=10, filename="/tmp/test_twinkle.gif"
    )
