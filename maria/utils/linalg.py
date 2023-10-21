import numpy as np
import scipy as sp


def get_rotation_matrix_from_angle(a):
    S = np.atleast_1d(a)[..., None, None] * np.array([[0.0, 1.0], [-1.0, 0.0]])
    R = sp.linalg.expm(S)
    return R.reshape(*np.shape(a), 2, 2)


def optimize_area_minimizing_rotation_matrix(points):
    def log_dimension_ratio(a):
        trans_points = points @ get_rotation_matrix_from_angle(a).T
        log_ratio = np.log(trans_points[:, 0].ptp() / trans_points[:, 1].ptp())
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
