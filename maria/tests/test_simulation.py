import pytest

from maria import Simulation


def test_default_sim():
    sim = Simulation()
    tod = sim.run()
