from __future__ import annotations

import numpy as np

from maria import Simulation
from maria.units import Calibration


def test_brightness_temperature_to_spectral_flux_density_per_pixel():
    assert np.isclose(
        Calibration("K_RJ -> Jy/pixel", nu=90e9, res=np.radians(1 / 60))(1e0),
        21.0576123,
    )


def test_decalibration():

    sim = Simulation(
        instrument="MUSTANG-2",
        plan="ten_second_stare",
        site="green_bank",
        noise=True,
    )

    tod = sim.run()

    assert np.isclose(
        tod.signal.compute(), tod.to("K_RJ").to("pW").signal.compute()
    ).all()
