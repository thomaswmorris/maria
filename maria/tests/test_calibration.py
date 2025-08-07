from __future__ import annotations

import numpy as np
import pytest

import maria
from maria import all_bands, all_regions
from maria.calibration import Calibration
from maria.constants import T_CMB
from maria.spectrum import AtmosphericSpectrum


def test_brightness_temperature_to_spectral_flux_density_per_pixel():
    square_arcminute = np.radians(1 / 60) ** 2

    assert np.isclose(
        Calibration("K_RJ -> Jy/pixel", nu=90e9, pixel_area=square_arcminute)(1e0),
        21.0576123,
    )


# def test_involution():

#     sim = Simulation(
#         instrument="MUSTANG-2",
#         plan="five_second_stare",
#         site="green_bank",
#         noise=True,
#     )

#     tod = sim.run()

#     assert np.isclose(
#         tod.signal.compute(), tod.to("K_RJ").to("pW").signal.compute()
#     ).all()

n_tests = 8
test_regions = np.random.choice(a=all_regions, size=n_tests)
test_bands = np.random.choice(a=all_bands, size=n_tests)


@pytest.mark.parametrize(
    "region,band_name",
    zip(test_regions, test_bands),
)
def test_cmb_atmosphere_reversability(region, band_name):
    eps = 1e-4
    kwargs_shape = (5, 7, 9)  # a random shape to confuse things

    band = maria.get_band(band_name)
    polarized = np.random.choice([True, False])
    spectrum = AtmosphericSpectrum(region=region)

    calibration_kwargs = {
        "spectrum": spectrum,
        "zenith_pwv": np.random.uniform(size=kwargs_shape, low=0, high=50),
        "base_temperature": np.random.uniform(
            size=kwargs_shape,
            low=spectrum.side_base_temperature.min(),
            high=spectrum.side_base_temperature.max(),
        ),
        "elevation": np.radians(np.random.uniform(size=kwargs_shape, low=10, high=90)),
        "polarized": polarized,
    }

    # calibration_kwargs = {}

    P_lo = Calibration("K_b -> fW", band=band, **calibration_kwargs)(T_CMB - eps / 2)
    P_hi = Calibration("K_b -> fW", band=band, **calibration_kwargs)(T_CMB + eps / 2)
    T = Calibration("fW -> K_CMB", band=band, **calibration_kwargs)(P_hi - P_lo)

    assert np.allclose(T, eps, rtol=1e-6)
