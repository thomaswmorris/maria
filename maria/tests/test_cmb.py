import time

import pytest

import maria
from maria.cmb import generate_cmb


def test_generate_cmb():
    generate_cmb(nside=1024)
