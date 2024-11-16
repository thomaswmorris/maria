from __future__ import annotations

import pytest

import maria
from maria.atmosphere import AtmosphericSpectrum


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_spectrum_from_cache_refresh(region_name):
    spectrum = AtmosphericSpectrum(region=region_name, refresh_cache=True)


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_spectrum_from_cache(region_name):
    spectrum = AtmosphericSpectrum(region=region_name)

    spectrum.emission(nu=90, elevation=45, pwv=1.5)
