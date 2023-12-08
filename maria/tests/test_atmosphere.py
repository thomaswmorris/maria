import pytest

import maria
from maria.atmosphere import LinearAngularSimulation, SingleLayerSimulation


@pytest.mark.atmosphere
def test_single_layer_model():
    sim = SingleLayerSimulation(array="MUSTANG-2", pointing="daisy", site="GBT")
    tod = sim.run()


@pytest.mark.atmosphere
def test_linear_angular_model():
    sim = LinearAngularSimulation(array="MUSTANG-2", pointing="daisy", site="GBT")
    tod = sim.run()
