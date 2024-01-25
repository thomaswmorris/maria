import pytest

import maria


@pytest.mark.parametrize("instrument_name", maria.all_instruments)
def test_get_instrument(instrument_name):
    instrument = maria.get_instrument(instrument_name)


@pytest.mark.parametrize("pointing_name", maria.all_pointings)
def test_get_pointing(pointing_name):
    pointing = maria.get_pointing(pointing_name)


@pytest.mark.parametrize("site_name", maria.all_sites)
def test_get_site(site_name):
    site = maria.get_site(site_name)
