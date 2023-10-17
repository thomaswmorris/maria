import pytest

@pytest.mark.mock_obs
def test_we_observe():
    from maria import Simulation
    from maria import mappers
    import numpy as np

    sim = Simulation(

    # Mandatory minimal weither settings
    # ---------------------
    array     = 'MUSTANG-2',       # Array type
    pointing  = 'daisy',      # Scanning strategy 
    site      = 'GBT',             # Site
    atm_model = 'linear_angular',  # The atmospheric model, set to None if you want a noiseless observation.
    # atm_model = None,              # The atmospheric model, set to None if you want a noiseless observation.
    
    # True sky input
    # ---------------------
    map_file     = "maps/protocluster.fits",                     # Input files must be a fits file.
                                                                          # map_file can also be set to None if are only interested in the noise
    map_center   = (150., 10),                                             # RA & Dec in degree

    # Defeault Observational setup
    # ----------------------------
    integration_time = 600,          # seconds
    scan_center      = (150., 10),    # degrees
    pointing_frame   = "ra_dec",     # frame
    scan_radius      = 0.05,         # How large the scanning pattern is in degree

    # Additional inputs:
    # ----------------------
    weather_quantiles    = {'column_water_vapor' : 0.5},    # Weather conditions specific for that site
    map_units    = 'Jy/pixel',                      # Kelvin Rayleigh Jeans (K, defeault) or Jy/pixel 
    # map_inbright = -6e-6,                        # Linearly scale the map to have this peak value.
    map_res      = 0.1 / 1000,                      # degree, overwrites header information
    )


    tod = sim.run()

    mapper = mappers.BinMapper(map_height=np.radians(10/60),
                            map_width=np.radians(10/60),
                            map_res=np.radians(0.4/1000))
    mapper.add_tods(tod)
    mapper.run()

    mapper.save_maps("/tmp/test.fits")