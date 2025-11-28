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


@pytest.mark.parametrize("filename", all_maps)
def test_all_maps(filename):
    m = maria.map.load(filename=fetch(filename), width=0.1, center=(150, 10))  # noqa
    m.plot()

    plt.close("all")


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_map_io_units(map_path):
    m = maria.map.load(fetch(map_path))
    if "nu" not in m.dims:
        m = m.unsqueeze("nu", 150e9)
    assert np.allclose(m.to("K_RJ").to("Jy/pixel").to(m.units).data, m.data).compute()

    m.to("cK_RJ").to_hdf("/tmp/test_write_map.h5")
    new_m_hdf = maria.map.load("/tmp/test_write_map.h5").to(m.units)  # noqa

    assert np.allclose(new_m_hdf.data, m.data)
    assert np.allclose(new_m_hdf.resolution.arcsec, m.resolution.arcsec)

    if "fits" in map_path:
        m.to("cK_RJ").to_fits("/tmp/test_write_map.fits")
        new_m_fits = maria.map.load("/tmp/test_write_map.fits").to(m.units)  # noqa

        assert np.allclose(new_m_fits.data, m.data)
        assert np.allclose(new_m_fits.resolution.arcsec, m.resolution.arcsec)

    m.plot()
