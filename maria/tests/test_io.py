import pytest

from maria.io import fetch


@pytest.mark.parametrize("filename", ["cluster.fits"])
def test_maps_from_cache(filename):
    fetch(f"maps/{filename}")
