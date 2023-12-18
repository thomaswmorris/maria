import time

import pytest

import maria
from maria.sim import Simulation
from maria.weather import Weather


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_weather(region_name):
    weather = Weather(t=time.time(), region=region_name)
    print(f"pwv={weather.pwv:.03f}mm")
