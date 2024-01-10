import os

import pytest

from maria.tod import TOD

here, this_filename = os.path.split(__file__)


def test_tod_io_fits_mustang2():
    mustang2_tod_file = f"{here}/../../data/tods/sample_mustang2_tod_60s.fits"

    tod = TOD.from_fits(fname=mustang2_tod_file, format="mustang-2")
    tod.to_fits(fname="/tmp/test_save_mustang2.fits", format="mustang-2")
