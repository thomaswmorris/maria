import pytest

import maria
from maria.io import fetch


@pytest.mark.parametrize(
    "map_name", ["maps/cluster.fits", "maps/big_cluster.fits", "maps/galaxy.fits"]
)
def test_atmosphere(map_name):
    map_filename = fetch(map_name)

    m = maria.map.read_fits(
        filename=map_filename,
        index=0,
        nu=90,
        resolution=1 / 1024,
        center=(150, 10),
        units="Jy/pixel",
    )

    m.plot()
