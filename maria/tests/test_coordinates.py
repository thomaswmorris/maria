from __future__ import annotations

import numpy as np

from maria.coords import offsets_to_phi_theta, phi_theta_to_offsets


def test_offsets_transform():
    n = 256

    for cphi in np.random.uniform(low=0, high=2 * np.pi, size=5):
        for ctheta in np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=5):
            offsets = np.radians(
                np.random.uniform(low=-0.5, high=+0.5, size=(n, 2)),
            )

            _phitheta = offsets_to_phi_theta(offsets, cphi, ctheta)
            _offsets = phi_theta_to_offsets(_phitheta, cphi, ctheta)

            assert np.mean(np.square(offsets - _offsets)) < 1e-5
