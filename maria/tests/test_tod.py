from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from maria import Simulation
from maria.coords import Coordinates, unjitted_offsets_to_phi_theta
from maria.instrument import get_instrument
from maria.noise import generate_noise_with_knee
from maria.plan import get_plan
from maria.tod.tod import TOD

plt.close("all")


def test_tod_functions():
    n = 256

    time = 1.75e9 + np.arange(0, 600, 0.1)
    azim = np.radians(45) * np.ones(len(time))
    elev = np.radians(45) * np.ones(len(time))

    offsets = np.radians(1e1 * np.random.standard_normal(size=(n, 1, 2)))

    PT = unjitted_offsets_to_phi_theta(offsets, azim, elev)

    coords = Coordinates(phi=PT[..., 0], theta=PT[..., 1], t=time, frame="az_el")

    noise = generate_noise_with_knee(shape=coords.shape, sample_rate=1 / coords.timestep, knee=0.5)

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
        "remove_spline": {"knot_spacing": 0.5},
    }

    tod.process(config=pp_config)

    tod.twinkle(
        rate=2,
        max_frames=10,
        filename="/tmp/test_twinkle.gif",
    )


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


def test_tod_write_and_load():
    plan = get_plan(
        scan_pattern="daisy",  # scanning pattern
        scan_options={"radius": 2 / 60, "speed": 0.5 / 60},  # in degrees
        duration=600,  # integration time in seconds
        sample_rate=50,  # in Hz
        scan_center=(202.27211, 47.195277),  # position in the sky
        frame="ra_dec",
    )

    sim = Simulation(get_instrument("MUSTANG-2"), plan=plan)
    tod = sim.run()
    tod.to_fits("/tmp/sim_tod.fits")
    tod_loaded = TOD.from_fits("/tmp/sim_tod.fits", format="MUSTANG-2")
