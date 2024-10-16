import time

import pytest

import maria
from maria.atmosphere import Atmosphere, AtmosphericSpectrum, Weather
from maria.io import fetch


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_spectrum_from_cache(region_name):
    AtmosphericSpectrum(region=region_name, refresh_cache=True)
