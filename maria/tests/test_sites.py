from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import maria

plt.close("all")


@pytest.mark.parametrize("site_name", maria.all_sites)
def test_get_site(site_name):
    site = maria.get_site(site_name)
    print(site)


def test_site_plot():
    site = maria.get_site("cerro_toco")
    site.plot()
