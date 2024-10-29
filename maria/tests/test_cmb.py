import time

import pytest

import maria
from maria.cmb import generate_cmb


@pytest.mark.parametrize("nside", [256, 512, 1024, 2048, 4096])
def test_generate_cmb(nside):
    generate_cmb(nside=nside)
