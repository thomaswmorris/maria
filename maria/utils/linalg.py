import numpy as np
import scipy as sp


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