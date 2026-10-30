from __future__ import annotations

import maria
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

    min_rel_noise = 0.7
    max_rel_noise = 1.3
    if scaled_residuals.std() < min_rel_noise:
        raise RuntimeError(
            f"Noise residuals are too low ({scaled_residuals.std():.03f} < {min_rel_noise})",
        )

    if scaled_residuals.std() > max_rel_noise:
        raise RuntimeError(
            f"Noise residuals are too high ({scaled_residuals.std():.03f} > {max_rel_noise})",
        )


def test_per_detector_noise_levels():
    band = maria.Band(name="f150", center=150e9, width=30e9, NEP=1e-17, knee=0)
    NEP = np.geomspace(1e-17, 1e-16, 64)
    offsets = np.random.uniform(low=0, high=0.1, size=(64, 2))
    instrument = maria.get_instrument(
        array={"bands": [band], "primary_size": 5, "xi": offsets[:, 0], "eta": offsets[:, 1], "NEP": NEP}
    )

    sim = Simulation(instrument=instrument, plans="ten_second_zenith_stare", site="green_bank", noise=True)
    tod = sim.run()[0].to("pW")

    # the white noise in each sample should have a standard deviation of NEP * sqrt(fs)
    rel_noise = tod.noise.compute().std(axis=1) / (1e12 * tod.dets.NEP * np.sqrt(tod.fs))
    assert np.abs(rel_noise.mean() - 1) < 0.02

    weight = np.asarray(tod.weight)[:, 0]
    assert np.allclose(weight, (np.median(tod.dets.NEP) / tod.dets.NEP) ** 2)


def test_explicit_band_noise_levels():
    band = maria.Band(name="f150", center=150e9, width=30e9, NEP=1e-17)
    offsets = np.random.uniform(low=0, high=0.1, size=(64, 2))
    array = {"bands": [band], "primary_size": 5, "xi": offsets[:, 0], "eta": offsets[:, 1]}

    # a per-detector NEP equal to that of the band should give the same noise
    noise = []
    for NEP in [None, np.full(64, 1e-17)]:
        instrument = maria.get_instrument(array={**array, **({"NEP": NEP} if NEP is not None else {})})
        sim = Simulation(instrument=instrument, plans="ten_second_zenith_stare", site="green_bank", noise=True)
        np.random.seed(42)
        tod = sim.run()[0]
        noise.append(tod.noise.compute())
        assert np.all(np.asarray(tod.weight) == 1)

    assert np.array_equal(*noise)
