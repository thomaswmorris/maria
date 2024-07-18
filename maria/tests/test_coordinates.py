import numpy as np
import pytest

from maria.coords import dx_dy_to_phi_theta, phi_theta_to_dx_dy


def test_offsets_transform():
    OFFSETS_SIZE = 256

    for cphi in np.random.uniform(low=0, high=2 * np.pi, size=16):
        for ctheta in np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=16):
            dx = np.radians(
                60 * np.random.uniform(low=-0.5, high=+0.5, size=OFFSETS_SIZE)
            )
            dy = np.radians(
                60 * np.random.uniform(low=-0.5, high=+0.5, size=OFFSETS_SIZE)
            )

            _phi, _theta = dx_dy_to_phi_theta(dx, dy, cphi, ctheta)
            _dx, _dy = phi_theta_to_dx_dy(_phi, _theta, cphi, ctheta)

            assert (dx - _dx).std() < 1e-8
            assert (dy - _dy).std() < 1e-8
