import time

import pytest

import maria
from maria.atmosphere import Atmosphere, AtmosphericSpectrum, Weather
from maria.io import fetch


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_atmosphere(region_name):
    Atmosphere(region=region_name)


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_spectrum_from_cache(region_name):
    AtmosphericSpectrum(region=region_name, refresh_cache=True)


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_weather_from_cache(region_name):
    Weather(region=region_name, refresh_cache=True)


@pytest.mark.parametrize("filename", ["cluster.fits"])
def test_maps_from_cache(filename):
    fetch(f"maps/{filename}")
