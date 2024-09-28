import numpy as np

from maria.coords import Coordinates, dx_dy_to_phi_theta
from maria.tod.sim import generate_noise_with_knee
from maria.tod.tod import TOD


def test_TOD():
    n = 256

    time = 1.75e9 + np.arange(0, 600, 0.1)
    azim = np.radians(45) * np.ones(len(time))
    elev = np.radians(45) * np.ones(len(time))

    offsets = np.radians(1e1 * np.random.standard_normal(size=(n, 2)))

    AZIM, ELEV = dx_dy_to_phi_theta(*offsets.T[..., None], azim, elev)

    coords = Coordinates(phi=AZIM, theta=ELEV, time=time, frame="az_el")

    noise = generate_noise_with_knee(t=time, n=n, NEP=0.01, knee=0.5)

    tod = TOD(data=dict(noise=dict(data=noise)), coords=coords)
