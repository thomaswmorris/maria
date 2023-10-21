import time

import pytest

import maria


def test_weather():
    for region in maria.weather.supported_regions.index:
        weather = maria.weather.Weather(t=time.time(), region=region)

        print(region, weather.pwv)
