import numpy as np
import pytest

from maria import Simulation


@pytest.mark.noise
def test_linear_angular_model():
    sim = Simulation(
        instrument="MUSTANG-2",
        plan="daisy",
        site="green_bank",
        noise=True,
    )
    tod = sim.run()

    target_error = sim.instrument.dets.NEP / np.sqrt(sim.plan.duration)

    scaled_residuals = (
        tod.data.compute().mean(axis=1) / target_error
    )  # this is should be distributed as a zero-mean unit-variance Gaussian

    min_noise = 0.9
    max_noise = 1.1
    if scaled_residuals.std() < min_noise:
        raise RuntimeError(
            f"Noise residuals are too low ({scaled_residuals.std():.03f} < {min_noise})"
        )

    if scaled_residuals.std() > max_noise:
        raise RuntimeError(
            f"Noise residuals are too high ({scaled_residuals.std():.03f} > {max_noise})"
        )
