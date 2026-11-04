from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.instrument import Band
from maria.io import fetch
from maria.map import all_maps
from maria.mapping import BinMapper

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


def test_beam_area():
    fwhm = 0.05  # in degrees
    x = np.linspace(-0.5, 0.5, 513)
    X, Y = np.meshgrid(x, x)
    data = np.exp(-4 * np.log(2) * (X**2 + Y**2) / fwhm**2)

    m = maria.map.ProjectionMap(data=data, width=1e0, nu=150e9, beam=(fwhm, fwhm, 0), center=(0, -30), units="Jy/pixel")

    # a point source smoothed by the beam should peak at its total flux in Jy/beam
    rel_error = (np.nanmax(m.to("Jy/beam").data) / np.nansum(m.data)).compute() - 1
    assert np.abs(rel_error) < 1e-3
