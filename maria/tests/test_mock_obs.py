import pytest

@pytest.mark.mock_obs
def test_we_observe():

    from maria.mock_obs import WeObserve

    for file in ["./maps/tsz.fits"]:
        obs = WeObserve(
            array_name='MUSTANG-2',
            pointing_name='DAISY_5deg_45az_45el_60s',
            site_name='GBT',
            project="./Mock_obs",
            skymodel=file,
            verbose=True,
            bands=[(27e9, 5e9, 100)],  # (band center, bandwidth, dets per band) [GHz, GHz, .]
            cmb = False,
            inbright=-5.37 * 1e3 * 0.000113,
            incell=0.5 / 360,  # degree
        )
