import numpy as np
import pytest

from maria.weather import Weather


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_weather_from_cache(region_name):
    weather = Weather(region=region_name, refresh_cache=True)

    weather.pwv
