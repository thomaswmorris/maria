import pytest
import maria
from maria.noise import NoiseSimulation


@pytest.mark.noise
def test_linear_angular_model():

    sim = NoiseSimulation(array="MUSTANG-2", pointing="daisy", site="GBT")
    tod = sim.run()


