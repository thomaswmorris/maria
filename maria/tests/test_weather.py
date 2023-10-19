import pytest
import maria
import time

def test_weather():

    for region in maria.weather.regions.index:

        weather = maria.weather.Weather(t=time.time(), region=region)

        print(region, weather.pwv)
