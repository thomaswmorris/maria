from __future__ import annotations

import pytest

import os
import maria
from maria.io import fetch


@pytest.mark.parametrize("filename", ["big_cluster.fits", "cluster.fits"])
def test_maps_from_cache(filename):
    map_filename = fetch(f"maps/{filename}")
    input_map = maria.map.read_fits(filename=map_filename, width=0.1, center=(150, 10))


def test_change_cache_dir():
    try:
        maria.set_cache_dir("/tmp/maria-data-2")
        fetch(f"maps/cluster.fits")
        assert os.path.isfile(os.environ["MARIA_CACHE_DIR"] + "/maps/cluster.fits")
    except:
        del os.environ["MARIA_CACHE_DIR"]
        raise Exception()
