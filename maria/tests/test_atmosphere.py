import time

import pytest

import maria
from maria import Simulation
from maria.atmosphere import Atmosphere, AtmosphericSpectrum, Weather


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_atmosphere(region_name):
    atmosphere = Atmosphere(region=region_name)


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_spectrum_from_cache(region_name):
    spectrum = AtmosphericSpectrum(region=region_name, refresh_cache=True)


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_weather_from_cache(region_name):
    weather = Weather(region=region_name, refresh_cache=True)


def test_atmosphere_2d():
    sim = Simulation(
        instrument="MUSTANG-2",
        plan="one_minute_zenith_stare",
        site="green_bank",
        atmosphere="2d",
    )
    tod = sim.run()
