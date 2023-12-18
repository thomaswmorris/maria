import pytest

from maria import Simulation


@pytest.mark.atmosphere
def test_linear_angular_model():
    sim = Simulation(
        array="MUSTANG-2", pointing="daisy", site="GBT", atmosphere_model="2d"
    )
    tod = sim.run()
