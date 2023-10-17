import pytest
import maria
from maria.atmosphere import SingleLayerSimulation


@pytest.mark.atmosphere
def test_linear_angular_model():

    sim = SingleLayerSimulation(array="MUSTANG-2", pointing="daisy", site="GBT")
    tod = sim.run()


