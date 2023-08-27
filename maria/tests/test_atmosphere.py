import pytest
import maria
from maria.atmosphere import LinearAngularSimulation


@pytest.mark.atmosphere
def test_linear_angular_model():

    sim = LinearAngularSimulation(array="MUSTANG-2", pointing="daisy", site="GBT", atm_model="linear_angular")
    tod = sim.run()


