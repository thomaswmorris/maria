import pytest

import maria
from maria.atmosphere import Weather


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_weather(region_name):
    weather = Weather(region=region_name)
    print(f"pwv={weather.pwv:.03f}mm")


@pytest.mark.parametrize("region_name", ["chajnantor", "green_bank", "south_pole"])
def test_weather_from_cache(region_name):
    weather = Weather(region=region_name, from_cache=True)
