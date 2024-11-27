from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from maria import Simulation
from maria.coords import Coordinates, dx_dy_to_phi_theta
from maria.noise import generate_noise_with_knee
from maria.tod.tod import TOD

plt.close("all")


def test_tod_functions():
    n = 256

    time = 1.75e9 + np.arange(0, 600, 0.1)
    azim = np.radians(45) * np.ones(len(time))
    elev = np.radians(45) * np.ones(len(time))

    offsets = np.radians(1e1 * np.random.standard_normal(size=(n, 2)))

    AZIM, ELEV = dx_dy_to_phi_theta(*offsets.T[..., None], azim, elev)

    coords = Coordinates(phi=AZIM, theta=ELEV, time=time, frame="az_el")

    noise = generate_noise_with_knee(t=time, n=n, NEP=0.01, knee=0.5)

    tod = TOD(data=dict(noise=noise), coords=coords)

    print(f"{tod.sample_rate =}")
    print(f"{tod.duration =}")
    print(f"{tod.boresight =}")


def test_tod_preprocessing_with_config():
    sim = Simulation()

    tod = sim.run()

    pp_config = {
        "window": {"name": "tukey", "kwargs": {"alpha": 0.25}},
        "filter": {"f_lower": 0.5},
        "remove_modes": {"modes_to_remove": [0]},
        "despline": {"knot_spacing": 0.5},
    }

    tod.process(config=pp_config)


def test_tod_preprocessing_with_kwargs():
    sim = Simulation()

    tod = sim.run()

    tod.process(window="tukey")


def test_tod_preprocessing_errors():
    sim = Simulation()

    tod = sim.run()

    try:
        tod.process(f_lower="a")
        assert False
    except TypeError:
        pass
