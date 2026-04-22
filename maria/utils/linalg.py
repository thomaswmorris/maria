from __future__ import annotations

import numpy as np
import scipy as sp


def compute_pointing_matrix_sparse_indices(x_list, bins_list):
    """
    Compute the pointing matrix for a set of points onto a Cartesian product of bins
    """

    n_samples = len(x_list[0].ravel())

    if not np.all([np.all(np.diff(bins) > 0) for bins in bins_list]):
        raise ValueError(f"Each set of bins must be strictly increasing")

    map_pixel_index = 0
    mask = np.ones_like(x_list[0].ravel(), dtype=bool)
    cum_npix = 1

    # indices_per_bin = np.zeros(())

    for dim, (x, bins) in enumerate(zip(x_list, bins_list)):
        dim_bins = np.digitize(x.ravel(), bins=bins)

        # print(dim_bins.min(), dim_bins.max(), len(bins))

        mask *= np.where((dim_bins > 0) & (dim_bins < len(bins)), True, False)
        map_pixel_index += cum_npix * (dim_bins - 1)
        cum_npix *= len(bins) - 1

    if not mask.sum():
        return [], [], cum_npix

    if map_pixel_index[mask].max() >= cum_npix:
        raise RuntimeError()

    return np.arange(n_samples)[mask], map_pixel_index[mask], cum_npix


def fast_psd_inverse(M):
    """
    About twice as fast as np.linalg.inv for large, PSD matrices.
    """

    cholesky, dpotrf_info = sp.linalg.lapack.dpotrf(M)
    invM, dpotri_info = sp.linalg.lapack.dpotri(cholesky)
    return np.where(invM, invM, invM.T)
