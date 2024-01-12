import pytest

from maria import Simulation


@pytest.mark.sim
def test_default_sim():
    sim = Simulation()
    tod = sim.run()
