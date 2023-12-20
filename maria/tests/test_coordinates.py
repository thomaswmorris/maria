import numpy as np
import pytest

from ..utils.coords import dx_dy_to_phi_theta, phi_theta_to_dx_dy


def test_offsets_transform():
    # test that the inverse works
    center_phi, center_theta = 3 * np.pi / 4, np.pi / 3
    phi, theta = dx_dy_to_phi_theta(0, np.radians(1), np.radians(130), np.radians(40))
    assert np.isclose(np.degrees(phi), 130) and np.isclose(np.degrees(theta), 41)

    # test that the inverse works
    in_dx, in_dy = 1e-2 * np.random.standard_normal(size=(2, 100))
    phi, theta = dx_dy_to_phi_theta(in_dx, in_dy, center_phi, center_theta)
    out_dx, out_dy = phi_theta_to_dx_dy(phi, theta, center_phi, center_theta)
    assert np.isclose(in_dx, out_dx, atol=1e-6).all()
    assert np.isclose(in_dy, out_dy, atol=1e-6).all()
