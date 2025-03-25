from __future__ import annotations

import numpy as np
import pytest

import maria
from maria import Simulation
from maria.atmosphere import Atmosphere, AtmosphericSpectrum


@pytest.mark.parametrize("region_name", maria.all_regions)
def test_atmosphere(region_name):
    atmosphere = Atmosphere(region=region_name)
    atmosphere.spectrum.emission(nu=90e9, elevation=1.1, pwv=1.5)


@pytest.mark.parametrize("region_name", ["chajnantor"])
def test_spectrum_from_cache(region_name):
    spectrum = AtmosphericSpectrum(region=region_name, refresh_cache=True)  # noqa


def test_atmosphere_2d():
    sim = Simulation(
        instrument="MUSTANG-2",
        plan="one_minute_zenith_stare",
        site="green_bank",
        atmosphere="2d",
    )
    tod = sim.run()  # noqa
