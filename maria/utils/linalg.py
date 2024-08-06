import numpy as np
import scipy as sp


def extrude(
    values: np.array,
    A: np.array,
    B: np.array,
    n_steps: int,
    n_i: int,
    n_j: int,
    i_sample_index: int,
    j_sample_index: int,
):
    # muy rapido
    BUFFER = np.zeros((n_steps + n_i) * n_j)
    BUFFER[n_steps * n_j :] = values

    # remember that (e, c) -> n_c * e + c
    for buffer_index in np.arange(n_steps)[::-1]:
        BUFFER[buffer_index * n_j + np.arange(n_j)] = A @ BUFFER[
            n_j * (buffer_index + 1 + i_sample_index) + j_sample_index
        ] + B @ np.random.standard_normal(size=n_j)

    return BUFFER[: n_steps * n_j]


def get_rotation_matrix_2d(a):
    S = np.atleast_1d(a)[..., None, None] * np.array([[0.0, 1.0], [-1.0, 0.0]])
    R = sp.linalg.expm(S)
    return R.reshape(*np.shape(a), 2, 2)


def get_rotation_matrix_3d(angles, axis=0):
    shaped_angles = np.atleast_1d(angles)
    R = np.ones(shaped_angles.shape)[..., None, None] * np.eye(3)
    j = [i for i in [0, 1, 2] if not i == axis]
    j0, j1 = np.meshgrid(j, j)
    R[..., j0, j1] = sp.linalg.expm(
        np.array([[0, -1], [+1, 0]]) * shaped_angles[..., None, None]
    )
    return R.reshape(*np.shape(angles), 3, 3)


def compute_optimal_rotation(points):
    if points.ndim != 2:
        raise ValueError("'points' must be an (n_dim, n_points) array.")

    d = len(points)
    i, j = np.triu_indices(n=d, k=1)

    def rotation_matrix_from_skew_entries(s):
        S = np.zeros((d, d))
        S[i, j] = s
        return sp.linalg.expm(S + -S.T)

    def loss(x, *args):
        R = rotation_matrix_from_skew_entries(x)
        return sum(
            np.log(np.ptp(R @ args[0], axis=1)) * np.array([-1, *np.ones(d - 1)])
        )

    res = sp.optimize.minimize(
        loss, x0=np.zeros(int(d * (d - 1) / 2)), args=points, tol=1e-10, method="SLSQP"
    )

    if not res.success:
        raise RuntimeError("Could not find optimal rotation.")

    return rotation_matrix_from_skew_entries(res.x)


def optimize_area_minimizing_rotation_matrix(points):
    def log_dimension_ratio(a):
        trans_points = points @ get_rotation_matrix_2d(a).T
        log_ratio = np.log(np.ptp(trans_points[:, 0]) / np.ptp(trans_points[:, 1]))
        return log_ratio

    test_angles = np.linspace(0, np.pi, 64)[:-1]
    test_ratios = list(map(log_dimension_ratio, test_angles))

    res = sp.optimize.minimize(
        lambda p: log_dimension_ratio(p[0]),
        x0=[test_angles[np.argmin(test_ratios)]],
        bounds=[(0, np.pi)],
        method="Nelder-Mead",
    )

    return res


def mprod(*M):
    if not len(M) > 0:
        raise ValueError("You must specify at least one matrix!")
    res = M[0]
    for M_ in M[1:]:
        res = np.matmul(res, M_)
    return res


def fast_psd_inverse(M):
    """
    About twice as fast as np.linalg.inv for large, PSD matrices.
    """

    cholesky, dpotrf_info = sp.linalg.lapack.dpotrf(M)
    invM, dpotri_info = sp.linalg.lapack.dpotri(cholesky)
    return np.where(invM, invM, invM.T)
