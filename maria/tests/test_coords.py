import numpy as np
import scipy as sp

from maria.coords import Coordinates


def test_coords():
    t = np.arange(0, 1800, 1e-1)
    a = np.radians(30) * sp.signal.sawtooth(2 * np.pi * t / 60, width=0.5)
    e = np.radians(45) * np.ones(len(t))

    coords = Coordinates(phi=a, theta=e, time=t)

    print(coords.az)
    print(coords.ra)


# def test_offsets_transform():
#     OFFSETS_SIZE = 256

#     for cphi in np.random.uniform(low=0, high=2 * np.pi, size=16):
#         for ctheta in np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=16):
#             dx = np.radians(
#                 60 * np.random.uniform(low=-0.5, high=+0.5, size=OFFSETS_SIZE)
#             )
#             dy = np.radians(
#                 60 * np.random.uniform(low=-0.5, high=+0.5, size=OFFSETS_SIZE)
#             )

#             _phi, _theta = dx_dy_to_phi_theta(dx, dy, cphi, ctheta)
#             _dx, _dy = phi_theta_to_dx_dy(_phi, _theta, cphi, ctheta)

#             assert (dx - _dx).std() < 1e-8
#             assert (dy - _dy).std() < 1e-8
