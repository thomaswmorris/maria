import pytest

from maria.noise import WhiteNoiseSimulation


@pytest.mark.noise
def test_linear_angular_model():
    sim = WhiteNoiseSimulation(
        array="MUSTANG-2", pointing="daisy", site="GBT", white_noise_level=1e0
    )
    tod = sim.run()
