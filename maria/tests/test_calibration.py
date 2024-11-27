from __future__ import annotations

import numpy as np
from maria.units import Calibration


def test_calibration():
    assert np.isclose(
        Calibration("K_RJ -> Jy/pixel", nu=90e9, res=np.radians(1 / 60))(1e0),
        21.0576123,
    )
