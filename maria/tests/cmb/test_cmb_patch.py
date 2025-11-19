from __future__ import annotations

import maria
import pytest


def test_cmb_patch():
    plan = maria.Plan.generate(
        scan_pattern="daisy",
        scan_options={"radius": 5, "speed": 1},  # in degrees
        duration=120,  # in seconds
        sample_rate=50,  # in Hz
        scan_center=(150, 50),
        jitter=0,
        frame="az/el",
    )

    cmb_patch = maria.cmb.generate_cmb_patch(width=15, center=plan.center())

    sim = maria.Simulation(
        instrument="test/1deg",
        plans=plan,
        site="cerro_toco",
        map=cmb_patch,
        noise=False,
    )

    tod = sim.run()[0].to("K_CMB")

    # should be around 110 uK
    cmb_rms_K = tod.signal.std(axis=-1).compute()

    assert all(cmb_rms_K > 0.5 * cmb_patch.rms)
    assert all(cmb_rms_K < 2.0 * cmb_patch.rms)
