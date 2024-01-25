import pytest

from maria.noise import NoiseSimulation


@pytest.mark.noise
def test_linear_angular_model():
    noise_sim = NoiseSimulation(
        instrument="MUSTANG-2", pointing="daisy", site="green_bank", white_noise_level=1e0
    )
    tod = noise_sim.run()
