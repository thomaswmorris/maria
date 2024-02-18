import time

import pytest

import maria
from maria import Simulation
from maria.atmosphere import Atmosphere


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_atmosphere(region_name):
    atmosphere = Atmosphere(region=region_name)


@pytest.mark.parametrize("region_name", ["chajnantor", "green_bank", "south_pole"])
def test_atmosphere_from_cache(region_name):
    atmosphere = Atmosphere(
        region=region_name, spectrum_from_cache=True, weather_from_cache=True
    )


def test_atmosphere_2d():
    sim = Simulation(
        instrument="MUSTANG-2",
        pointing="daisy",
        site="green_bank",
        atmosphere_model="2d",
    )
    tod = sim.run()
