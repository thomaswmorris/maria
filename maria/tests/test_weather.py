from __future__ import annotations

import numpy as np
import pytest

import maria
from maria.weather import Weather


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_weather_from_cache(region_name):
    weather = Weather(region=region_name)

    print(f"{weather.pwv =}")


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_weather_from_cache_refresh(region_name):
    weather = Weather(region=region_name, refresh_cache=True)  # noqa

    h = weather.base_altitude + np.linspace(0, 1000, 10)

    for k, v in weather(h).items():
        assert not np.isnan(v).any()
