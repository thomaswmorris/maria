from __future__ import annotations

import os

import pytest

import maria
from maria.io import fetch


def test_change_cache_dir():
    try:
        maria.set_cache_dir("/tmp/maria-data-2")
        fetch("maps/cluster.fits")
        assert os.path.isfile(os.environ["MARIA_CACHE_DIR"] + "/maps/cluster.fits")
    except Exception:
        del os.environ["MARIA_CACHE_DIR"]
        raise Exception()


@pytest.mark.parametrize("filename", ["big_cluster.h5", "cluster.fits"])
def test_maps_from_cache(filename):
    map_filename = fetch(f"maps/{filename}")
    m = maria.map.load(filename=map_filename, width=0.1, center=(150, 10))  # noqa
