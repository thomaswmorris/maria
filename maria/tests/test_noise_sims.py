import pytest
import maria
from maria.noise import NoiseSimulation


@pytest.mark.noise
def test_linear_angular_model():

    sim = NoiseSimulation(array="default", pointing="default", site="default")
    tod = sim.run()


