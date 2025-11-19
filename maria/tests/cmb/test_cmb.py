from __future__ import annotations

import maria
import pytest


@pytest.mark.parametrize("nside", [256, 512, 1024, 2048])
def test_generate_cmb(nside):
    cmb = maria.cmb.generate_cmb(nside=nside)
    # cmb.plot()


def test_cmb_calibration():
    plan = maria.Plan.generate(
        scan_pattern="daisy",
        scan_options={"radius": 10, "speed": 1},  # in degrees
        duration=120,  # in seconds
        sample_rate=50,  # in Hz
        scan_center=(150, 50),
        jitter=0,
        frame="az/el",
    )

    sim = maria.Simulation(
        instrument="test/1deg",
        plans=plan,
        site="cerro_toco",
        cmb="generate",
        noise=False,
    )

    tod = sim.run()[0].to("uK_CMB")

    # should be around 110 uK
    cmb_rms_uK = tod.signal.std(axis=-1).compute()

    assert all(cmb_rms_uK > 80)
    assert all(cmb_rms_uK < 150)
