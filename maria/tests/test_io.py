from __future__ import annotations

import os

import pytest

import maria
from maria import all_maps
from maria.io import fetch


def test_change_cache_dir():
    try:
        maria.set_cache_dir("/tmp/maria-data-2")
        fetch("maps/cluster1.fits")
        assert os.path.isfile(os.environ["MARIA_CACHE_DIR"] + "/maps/cluster1.fits")
        maria.set_cache_dir("/tmp/maria-data")
    except Exception:
        del os.environ["MARIA_CACHE_DIR"]
        maria.set_cache_dir("/tmp/maria-data")
        raise Exception()


@pytest.mark.parametrize("filename", all_maps)
def test_all_maps(filename):
    m = maria.map.load(filename=fetch(filename), width=0.1, center=(150, 10))  # noqa
    m.plot()
