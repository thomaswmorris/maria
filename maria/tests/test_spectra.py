import time

import pytest

import maria
from maria.atmosphere import AtmosphericSpectrum


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_spectra(region_name):
    spectrum = AtmosphericSpectrum(region=region_name)


@pytest.mark.parametrize("region_name", ["chajnantor", "green_bank", "south_pole"])
def test_spectra_from_cache(region_name):
    spectrum = AtmosphericSpectrum(region=region_name, from_cache=True)
