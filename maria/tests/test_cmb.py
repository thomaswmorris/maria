from __future__ import annotations

import pytest
import maria


@pytest.mark.parametrize("nside", [256, 512, 1024, 2048, 4096])
def test_generate_cmb(nside):

    cmb = maria.cmb.generate_cmb(nside=nside)
    cmb.plot()


def test_cmb_calibration():

    plan = maria.Plan(
        scan_pattern="daisy",
        scan_options={"radius": 5, "speed": 1},  # in degrees
        duration=60,  # in seconds
        sample_rate=50,  # in Hz
        scan_center=(150, 10),
        jitter=0,
        frame="ra_dec",
    )

    sim = maria.Simulation(
        instrument="test/1deg",
        plan=plan,
        site="cerro_toco",
        cmb="generate",
        noise=False,
    )

    tod = sim.run().to("uK_CMB")

    cmb_rms_uK = tod.signal.std(axis=-1).compute()

    assert 100 < cmb_rms_uK.mean() < 120
