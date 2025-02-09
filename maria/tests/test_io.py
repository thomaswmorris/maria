from __future__ import annotations

import pytest

import os
import maria
import numpy as np

from maria.io import fetch


def test_change_cache_dir():
    try:
        maria.set_cache_dir("/tmp/maria-data-2")
        fetch(f"maps/cluster.fits")
        assert os.path.isfile(os.environ["MARIA_CACHE_DIR"] + "/maps/cluster.fits")
    except:
        del os.environ["MARIA_CACHE_DIR"]
        raise Exception()


@pytest.mark.parametrize("filename", ["big_cluster.fits", "cluster.fits"])
def test_maps_from_cache(filename):
    map_filename = fetch(f"maps/{filename}")
    m = maria.map.read_fits(filename=map_filename, width=0.1, center=(150, 10))


@pytest.mark.parametrize("filename", ["big_cluster.fits", "cluster.fits"])
def test_maps_io(filename):

    map_filename = fetch(f"maps/{filename}")
    m = maria.map.load(filename=map_filename, width=1e0, units="K_RJ")

    m.to("cK_RJ").to_fits("/tmp/test.fits")

    new_m = maria.map.load("/tmp/test.fits").to("MK_RJ")
    new_m.to_hdf("/tmp/test.h5")

    new_new_m = maria.map.load("/tmp/test.h5").to("K_RJ")
