from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.plan import scan_types
from maria.plan.patterns import parse_scan_kwargs


@pytest.mark.parametrize("scan_type", scan_types.index)
def test_pattern(scan_type):
    plan = maria.Plan.generate(scan_type=scan_type)
    print(plan)

    plan.plot()
    plan.plot_hits()

    plt.close("all")


@pytest.mark.parametrize("scan_type", scan_types.index)
def test_pattern_speed(scan_type):
    #     plan = maria.Plan.generate(scan_type=scan_type)
    #     print(plan)

    #     plan.plot()
    #     plan.plot_hits()

    #     from maria.plan import scan_types

    time = np.arange(0, 3600, 0.01)

    # for index, entry in scan_types.iterrows():

    if scan_type in ["stare"]:
        return

    for trial in range(16):
        scan_kwargs = {
            "x_throw": np.random.choice(np.geomspace(1e-1, 1e0, 256)),  # in degrees
            "speed": np.random.choice(np.geomspace(1e-1, 1e0, 256)),  # in degrees
        }

        x, y = scan_types.loc[scan_type].generator(time, **parse_scan_kwargs(scan_kwargs)).T
        vx = np.diff(x) / np.diff(time)
        vy = np.diff(y) / np.diff(time)

        max_speed = np.sqrt(vx**2 + vy**2).max()

        assert np.isclose(max_speed, scan_kwargs["speed"], rtol=2e-1)
