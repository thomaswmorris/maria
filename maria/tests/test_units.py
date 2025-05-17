from __future__ import annotations

import pytest

import maria
from maria.units import Quantity


def test_units():
    assert str(Quantity(0.1, "deg")) == "6’"
    assert str(Quantity(1e-8, "deg")) == "36 uarcsec"
    assert str(Quantity(1e6, "uK_RJ")) == "1 K_RJ"
    assert str(Quantity(1e15, "K_RJ")) == "1e+15 K_RJ"
    assert str(Quantity(1e6, "m")) == "1000 km"
    assert str(Quantity(1000, "s")) == "1000 s"
