from maria.mock_obs import WeObserve

def test_we_observe():

    for file in ["./maps/tsz.fits"]:
        obs = WeObserve(
            array_name='AtLAST',
            pointing_name='DAISY_2deg_4ra_10.5dec_600s',
            site_name='APEX',
            project="./Mock_obs",
            skymodel=file,
            verbose=True,
            bands=[(27e9, 5e9, 100)],  # (band center, bandwidth, dets per band) [GHz, GHz, .]
            inbright=-5.37 * 1e3 * 0.00011347448463627645,
            incell=0.5 / 360,  # degree
        )
