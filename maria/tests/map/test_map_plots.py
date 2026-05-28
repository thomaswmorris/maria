from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import pytest
from maria.io import fetch
from maria.map import all_maps

plt.close("all")


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_plot_all_map_slices(map_path):

    m = maria.map.load(fetch(map_path))

    try:
        m.plot(slices="all")
    except Exception:
        if not sum([n > 1 for n in m.slice_dims.values()]) > 2:
            assert False
