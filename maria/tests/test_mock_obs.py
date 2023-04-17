import pytest

@pytest.mark.mock_obs
def test_we_observe():

    from maria.mock_obs import WeObserve

    for file in ["./maps/tsz.fits"]:
        obs = WeObserve(
            array_name    = 'MUSTANG-2',
            pointing_name = 'DAISY_2deg',
            site_name     = 'GBT',
            project       = './Mock_obs',
            skymodel      = file,

            integration_time = 600,       # seconds
            coord_center     = [4, 10.5], # degree
            coord_frame      = "ra_dec",

            # --- Additional
            verbose       = True,
            cmb           = False,

            bands = [(27e9, 5e9, 100),
                     (35e9, 5e9, 100)],  # (band center, bandwidth, dets per band) [GHz, GHz, .]
                     
            units     = 'Jy/pixel',                 # Kelvin Rayleigh Jeans (KRJ) or Jy/pixel            
            inbright  = -5.37 * 1e3 * 0.000113,     # In units of key units 
            incell    = 0.5 / 360,                  # degree
            quantiles = {'column_water_vapor':0.5}  # pwv = 50%
        )
