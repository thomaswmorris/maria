import time

import pytest

import maria


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_weather(region_name):
    weather = maria.weather.Weather(t=time.time(), region=region_name)
    print(f"pwv={weather.pwv:.03f}mm")
