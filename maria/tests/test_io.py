import pytest

import maria
from maria.io import fetch


@pytest.mark.parametrize("filename", ["big_cluster.fits", "cluster.fits"])
def test_maps_from_cache(filename):
    map_filename = fetch(f"maps/{filename}")

    input_map = maria.map.read_fits(filename=map_filename, width=0.1, center=(150, 10))
