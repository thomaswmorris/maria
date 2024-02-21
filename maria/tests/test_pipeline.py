import os

import numpy as np
import pytest

from maria import Simulation
from maria.map.mappers import BinMapper

from ..utils.io import fetch_cache

here, this_filename = os.path.split(__file__)

TEST_MAP_URL = (
    "https://github.com/thomaswmorris/maria-data/raw/master/maps/cluster.fits"
)


@pytest.mark.mock_obs
def test_mustang2():
    fetch_cache(TEST_MAP_URL, "/tmp/test_map.fits", refresh=True)
    map_size = 0.1

    pointing_center = (73.5287496858916, 2.961663679507145)
    pixel_size = 8.71452898559111e-05
    duration = 1 * 60.0
    sample_rate = 100
    scan_velocity = 38 / 3600

    inputfile = "/tmp/test_map.fits"
    outfile_map = "/tmp/test_map_output.fits"

    atm_model = "2d"
    white_noise_level = 1.3e-2
    pink_noise_level = 2.4

    sim = Simulation(
        # Mandatory minimal weither settings
        # ---------------------
        instrument="MUSTANG-2",  # Instrument type
        site="green_bank",  # Site
        plan="daisy",  # Scanning strategy
        atmosphere_model=atm_model,  # atmospheric model
        white_noise_level=white_noise_level,  # white noise level
        pink_noise_level=pink_noise_level,  # pink noise level
        # True sky input
        # ---------------------
        map_file=inputfile,  # Input files must be a fits file.
        map_units="Jy/pixel",  # Units of the input map in Kelvin Rayleigh Jeans (K, defeault) or Jy/pixel
        map_res=pixel_size,  # resolution of the map
        map_center=pointing_center,  # RA & Dec in degree
        map_freqs=[93],
        dets={"f093": {"n": 217, "bands": ["mustang2/f093"]}},
        # MUSTANG-2 Observational setup
        # ----------------------------s
        scan_options={
            "radius": 4.0 / 60.0,  # The radius of the Daisy scan in degrees
            "speed": scan_velocity,  # scan velocity in when the scan goes through the center deg/s
        },
        duration=duration,  # Seconds
        sample_rate=sample_rate,  # Hz
        scan_center=pointing_center,  # Degrees
        pointing_frame="ra_dec",  # Frame
        start_time="2022-02-11T23:00:00",  # observation date
        pwv_rms_frac=0.005,  # level of atmospheric fluctuations
    )

    tod = sim.run()

    mapper = BinMapper(
        center=(tod.coords.center_ra, tod.coords.center_dec),
        frame="ra_dec",
        width=np.radians(10.0 / 60.0),
        height=np.radians(10.0 / 60.0),
        res=np.radians(4.0 / 3600.0),
        degrees=False,
        tod_postprocessing={
            "remove_modes": {"n": 1},
            "filter": {"f": 0.08},
            "despline": {"spacing": 10},
        },
        map_postprocessing={
            "gaussian_filter": {"sigma": 1},
            "median_filter": {"size": 1},
        },
        tods=[tod],
    )

    mapper.run()

    mapper.save_maps(outfile_map)
