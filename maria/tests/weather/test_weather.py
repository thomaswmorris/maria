from __future__ import annotations

import maria
import numpy as np
import pytest
from maria.weather import Weather


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_weather_from_cache(region_name):
    weather = Weather(region=region_name)

    print(f"{weather.pwv =}")


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_weather_from_cache_refresh(region_name):
    weather = Weather(region=region_name, refresh_cache=True)  # noqa

    h = weather.altitude.m + np.linspace(0, 1000, 10)

    for k, v in weather(h, units=False).items():
        assert not np.isnan(v).any()
