from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.instrument import Band
from maria.io import fetch
from maria.map import all_maps
from maria.mappers import BinMapper

plt.close("all")


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_map_units(map_path):

    m = maria.map.load(fetch(map_path))
    if "nu" not in m.dims:
        m = m.unsqueeze("nu", 150e9)

    rel_error = np.nanstd(m.to("uK_RJ").to("Jy/pixel").to(m.units).data - m.data).compute() / np.nanstd(m.data).compute()
    assert rel_error < 1e-6
