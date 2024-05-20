import os

import numpy as np
import pytest

import maria
from maria import Simulation
from maria.map.mappers import BinMapper

from ..io import fetch_cache

here, this_filename = os.path.split(__file__)

TEST_MAP_SOURCE_URL = (
    "https://github.com/thomaswmorris/maria-data/raw/master/maps/cluster.fits"  # noqa
)
TEST_MAP_CACHE_PATH = "/tmp/maria-data/maps/test.fits"


@pytest.mark.mock_obs
def test_mustang2():
    fetch_cache(TEST_MAP_SOURCE_URL, TEST_MAP_CACHE_PATH)

    pointing_center = (73.5287496858916, 2.961663679507145)
    pixel_size = 8.71452898559111e-05
    duration = 1 * 60.0
    sample_rate = 100
    scan_velocity = 38 / 3600

    instrument = maria.get_instrument("MUSTANG-2")
    site = maria.get_site("green_bank")

    plan = maria.get_plan(
        "daisy",
        scan_options={
            "radius": 4.0 / 60.0,  # The radius of the Daisy scan in degrees
            "speed": scan_velocity,  # scan velocity in when the scan goes through the center deg/s
        },
        duration=duration,  # Seconds
        sample_rate=sample_rate,  # Hz
        scan_center=pointing_center,  # Degrees
        frame="ra_dec",  # Frame
        start_time="2022-02-11T23:00:00",  # observation date
    )

    map = maria.map.from_fits(
        filename=TEST_MAP_CACHE_PATH,
        resolution=pixel_size,
        frequency=90,
        center=pointing_center,
    )

    sim = Simulation(
        instrument=instrument,  # Instrument type
        site=site,  # Site
        plan=plan,  # Scanning strategy
        atmosphere="2d",  # atmospheric model
        map=map,
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

    mapper.save_maps("/tmp/test-output.fits")
