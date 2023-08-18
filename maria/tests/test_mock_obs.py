import pytest

@pytest.mark.mock_obs
def test_we_observe():
    from maria import Simulation
    from maria import mappers
    import numpy as np

    sim = Simulation(

        # Mandatory minimal weither settings
        # ---------------------
        array    = 'MUSTANG-2',
        pointing = 'DAISY_2deg',
        site     = 'GBT',
        detectors = [[150e9, 10e9, 750],
                     [220e9, 10e9, 750]],
        
        # True sky input
        # ---------------------
        map_file     = "./maps/protocluster.fits", #input files must be a fits file
        # map_file     = "/Users/jvanmarr/Documents/GitHub/maria/maps/ACT0329_CL0035.097.z_036.00GHz.fits", #input files must be a fits file
        map_center   = (4, 10.5), # RA & Dec in degree

        # Defeault Observational setup
        # ----------------------------
        integration_time = 600,         # seconds
        pointing_center = (4, 10.5),    # degrees
        pointing_frame  = "ra_dec",     # frame
        pointing_throws = (1., 1.),     # How large the scanning pattern is in degree

        # Additional inputs:
        # ----------------------
        quantiles    = {'column_water_vapor' : 0.5},  # Weather conditions specific for that site
        map_units    = 'Jy/pixel',                    # Kelvin Rayleigh Jeans (KRJ, defeault) or Jy/pixel 
        # map_inbright = -5.37 * 1e3 * 0.000113,        # In units of the map_units key
        map_res      = 0.5 / 1000,                    # degree, overwrites header information
    )

    tod = sim.run()

    mapper = mappers.RawBinMapper(resolution=np.radians(0.1/60))
    mapper.add_tods(tod)
    mapper.run()

    sim.save_maps(mapper)