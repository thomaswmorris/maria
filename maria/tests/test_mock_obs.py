import numpy as np
import pytest

from maria import Simulation, mappers


@pytest.mark.mock_obs
def test_sim():
    sim = Simulation(array="MUSTANG-2", pointing="daisy", site="GBT")
    tod = sim.run()


@pytest.mark.mock_obs
def test_sim_with_params():
    sim = Simulation(
        # Mandatory minimal weither settings
        # ---------------------
        array="MUSTANG-2",  # Array type
        pointing="daisy",  # Scanning strategy
        site="GBT",  # Site
        atm_model="single_layer",  # The atmospheric model, set to None if you want a noiseless observation.
        # True sky input
        # ---------------------
        map_file="maps/cluster.fits",  # Input files must be a fits file.
        # map_file can also be set to None if are only interested in the noise
        map_center=(150.0, 10),  # RA & Dec in degree
        # Defeault Observational setup
        # ----------------------------
        integration_time=600,  # seconds
        scan_center=(150.0, 10),  # degrees
        pointing_frame="ra_dec",  # frame
        scan_options={"radius": 0.05, "speed": 0.05, "petals": 5},
        # Additional inputs:
        # ----------------------
        map_units="Jy/pixel",  # Kelvin Rayleigh Jeans (K, defeault) or Jy/pixel
        # map_inbright = -6e-6,                        # Linearly scale the map to have this peak value.
        map_res=0.1 / 1000,  # degree, overwrites header information
    )

    tod = sim.run()

    mapper = mappers.BinMapper(
        map_height=np.radians(10 / 60),
        map_width=np.radians(10 / 60),
        map_res=np.radians(0.4 / 1000),
    )
    mapper.add_tods(tod)
    mapper.run()
