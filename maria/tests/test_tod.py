import os

import pytest

from maria.tod import TOD

from ..utils.io import fetch_cache

here, this_filename = os.path.split(__file__)

TEST_MUSTANG_TOD_URL = "https://github.com/thomaswmorris/maria-data/raw/master/tods/mustang2/m2_sample_tod_60s.fits"


def test_tod_io_fits_mustang2():
    fetch_cache(TEST_MUSTANG_TOD_URL, "/tmp/test_m2_tod.fits", refresh=True)
    tod = TOD.from_fits(fname="/tmp/test_m2_tod.fits", format="mustang-2")
    tod.to_fits(fname="/tmp/test_save_m2_tod.fits", format="mustang-2")
