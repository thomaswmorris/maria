from __future__ import annotations

import numpy as np
import pytest

from maria.units import Quantity


def test_repr():
    assert str(Quantity(0.1, "deg")) == "6’"
    assert str(Quantity(1e-8, "deg")) == "36 uarcsec"
    assert str(Quantity(1e6, "uK_RJ")) == "1 K_RJ"
    assert str(Quantity(1e15, "K_RJ")) == "1e+15 K_RJ"  # don't go high
    assert str(Quantity(1e-6, "Hz")) == "1e-06 Hz"  # don't go low
    assert str(Quantity(1000, "s")) == "1000 s"


def test_parsing():
    for nu in [
        (90e9, 150e9, 220e9),
        np.array([90e9, 150e9, 220e9]),
        [Quantity(90, "GHz"), Quantity(150, "GHz"), Quantity(220, "GHz")],
        Quantity((90e9, 150e9, 220e9), "Hz"),
    ]:
        q = Quantity(nu, "Hz")
        assert all(q.GHz == [90, 150, 220])
