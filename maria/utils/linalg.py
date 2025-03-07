from __future__ import annotations

import numpy as np
import scipy as sp

# def extrude(
#     values: np.array,
#     A: np.array,
#     B: np.array,
#     n_steps: int,
#     n_i: int,
#     n_j: int,
#     i_sample_index: int,
#     j_sample_index: int,
# ):
#     # muy rapido
#     BUFFER = np.zeros((n_steps + n_i) * n_j)
#     BUFFER[n_steps * n_j :] = values

#     # remember that (e, c) -> n_c * e + c
#     for buffer_index in np.arange(n_steps)[::-1]:
#         BUFFER[buffer_index * n_j + np.arange(n_j)] = A @ BUFFER[
#             n_j * (buffer_index + 1 + i_sample_index) + j_sample_index
#         ] + B @ np.random.standard_normal(size=n_j)

#     return BUFFER[: n_steps * n_j]


# def get_rotation_matrix_2d(a):
#     S = np.atleast_1d(a)[..., None, None] * np.array([[0.0, 1.0], [-1.0, 0.0]])
#     R = sp.linalg.expm(S)
#     return R.reshape(*np.shape(a), 2, 2)


# def get_rotation_matrix_3d(angles, axis=0):
#     shaped_angles = np.atleast_1d(angles)
#     R = np.ones(shaped_angles.shape)[..., None, None] * np.eye(3)
#     j = [i for i in [0, 1, 2] if not i == axis]
#     j0, j1 = np.meshgrid(j, j)
#     R[..., j0, j1] = sp.linalg.expm(
#         np.array([[0, -1], [+1, 0]]) * shaped_angles[..., None, None],
#     )
#     return R.reshape(*np.shape(angles), 3, 3)


def fast_psd_inverse(M):
    """
    About twice as fast as np.linalg.inv for large, PSD matrices.
    """

    cholesky, dpotrf_info = sp.linalg.lapack.dpotrf(M)
    invM, dpotri_info = sp.linalg.lapack.dpotri(cholesky)
    return np.where(invM, invM, invM.T)
