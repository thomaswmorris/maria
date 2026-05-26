from __future__ import annotations

import os

import maria
import numpy as np
import pytest
from maria import Plan, Simulation, all_instruments, all_sites, get_instrument
from maria.io import read_yaml
from maria.mappers import BinMapper

here, this_filename = os.path.split(__file__)

# all_test_plan_configs = list(read_yaml(f"{here}/../configs/test_plans.yml").values())


n_sims = 10
test_instruments = np.random.choice(a=[i for i in all_instruments if "alma" not in i], size=n_sims)
test_sites = np.random.choice(a=all_sites, size=n_sims)


@pytest.mark.parametrize(
    "instrument,site",
    zip(test_instruments, test_sites),
)
def test_tod_write_and_load(instrument, site):
    plan = Plan.generate(
        scan_pattern="daisy",  # scanning pattern
        scan_options={"radius": 2 / 60, "speed": 0.5 / 60},  # in degrees
        duration=60,  # integration time in seconds
        sample_rate=10,  # in Hz
        scan_center=(202.27211, 47.195277),  # position in the sky
        frame="az/el",
    )

    sim = Simulation(get_instrument(instrument), plans=plan, site=site)
    tod = sim.run()[0]
    tod.to_fits("/tmp/test_tod.fits")
    tod_loaded = maria.tod.load("/tmp/test_tod.fits", site=site, bands=sim.instrument.bands)
