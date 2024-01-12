import pytest

from maria import Simulation


@pytest.mark.atmosphere
def test_atmosphere_2d():
    sim = Simulation(
        array='MUSTANG-2', pointing='daisy', site='green_bank', atmosphere_model='2d'
    )
    tod = sim.run()
