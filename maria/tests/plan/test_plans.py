from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
from maria.plan import scan_patterns
from maria.plan.patterns import parse_scan_kwargs


@pytest.mark.parametrize("scan_pattern", scan_patterns.index)
def test_pattern(scan_pattern):
    plan = maria.Plan.generate(scan_pattern=scan_pattern)
    print(plan)

    plan.plot()
    plan.plot_counts()

    plt.close("all")


@pytest.mark.parametrize("scan_pattern", scan_patterns.index)
def test_pattern_speed(scan_pattern):
    #     plan = maria.Plan.generate(scan_pattern=scan_pattern)
    #     print(plan)

    #     plan.plot()
    #     plan.plot_counts()

    #     from maria.plan import scan_patterns

    time = np.arange(0, 3600, 0.01)

    # for index, entry in scan_patterns.iterrows():

    if scan_pattern in ["stare"]:
        return

    for trial in range(16):
        scan_kwargs = {
            "x_throw": np.random.choice(np.geomspace(1e-1, 1e0, 256)),  # in degrees
            "speed": np.random.choice(np.geomspace(1e-1, 1e0, 256)),  # in degrees
        }

        x, y = scan_patterns.loc[scan_pattern].generator(time, **parse_scan_kwargs(scan_kwargs))
        vx = np.diff(x) / np.diff(time)
        vy = np.diff(y) / np.diff(time)

        max_speed = np.sqrt(vx**2 + vy**2).max()

        assert np.isclose(max_speed, scan_kwargs["speed"], rtol=2e-1)
