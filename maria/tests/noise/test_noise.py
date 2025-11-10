from __future__ import annotations

import numpy as np
from maria import Simulation


def test_white_noise_levels():
    sim = Simulation(
        instrument="MUSTANG-2",
        plans="ten_second_zenith_stare",
        site="green_bank",
        noise=True,
    )
    tod = sim.run()[0].to("pW")

    target_error = sim.instrument.bands[0].NEP.to("pW√s") / np.sqrt(sim.plans.duration.s)

    # this is should be distributed as a zero-mean unit-variance Gaussian
    scaled_residuals = tod.noise.compute().mean(axis=1) / target_error

    min_rel_noise = 0.8
    max_rel_noise = 1.2
    if scaled_residuals.std() < min_rel_noise:
        raise RuntimeError(
            f"Noise residuals are too low ({scaled_residuals.std():.03f} < {min_rel_noise})",
        )

    if scaled_residuals.std() > max_rel_noise:
        raise RuntimeError(
            f"Noise residuals are too high ({scaled_residuals.std():.03f} > {max_rel_noise})",
        )
