from __future__ import annotations

import numpy as np
import pytest

import maria
from maria.plan import scan_patterns


@pytest.mark.parametrize("scan_pattern", scan_patterns.index)
def test_pattern(scan_pattern):
    plan = maria.Plan(scan_pattern=scan_pattern)
    print(plan)

    plan.plot()
    plan.plot_counts()


@pytest.mark.parametrize("scan_pattern", scan_patterns.index)
def test_pattern_speed(scan_pattern):
    #     plan = maria.Plan(scan_pattern=scan_pattern)
    #     print(plan)

    #     plan.plot()
    #     plan.plot_counts()

    #     from maria.plan import scan_patterns

    time = np.arange(0, 3600, 0.01)

    # for index, entry in scan_patterns.iterrows():

    if scan_pattern in ["stare"]:
        return

    for trial in range(16):
        radius = np.random.choice(np.geomspace(1e-1, 1e0, 256))  # in degrees
        speed = np.random.choice(np.geomspace(1e-1, 1e0, 256))  # in degrees

        x, y = scan_patterns.loc[scan_pattern].generator(time, radius=radius, speed=speed)
        vx = np.diff(x) / np.diff(time)
        vy = np.diff(y) / np.diff(time)

        max_speed = np.sqrt(vx**2 + vy**2).max()

        assert np.isclose(max_speed, speed, rtol=1e-1)
