import numpy as np
import pytest

from maria import Simulation
from maria.map.mappers import BinMapper


@pytest.mark.mock_obs
def test_sim():
    sim = Simulation(array="MUSTANG-2", pointing="daisy", site="GBT")
    tod = sim.run()


@pytest.mark.mock_obs
def test_sim_with_params():
    map_size = 0.1

    sim = Simulation(
        # Mandatory minimal weither settings
        # ---------------------
        array="MUSTANG-2",  # Array type
        pointing="daisy",  # Scanning strategy
        site="GBT",  # Site
        atmosphere_model="2d",  # The atmospheric model, set to None if you want a noiseless observation.
        # True sky input
        # ---------------------
        map_file="maps/cluster.fits",  # Input files must be a fits file.
        # map_file can also be set to None if are only interested in the noise
        map_center=(150.0, 10),  # RA & Dec in degree
        map_res=0.1 / 1000,  # degree, overwrites header information
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
    )

    tod = sim.run()

    mapper = BinMapper(
        center=(tod.boresight.center_ra, tod.boresight.center_dec),
        frame="ra_dec",
        width=map_size,
        height=map_size,
        res=map_size / 64,
        filter_data=True,
        n_modes_to_remove=1,
    )

    mapper.add_tods(tod)
    mapper.run()
