from __future__ import annotations

import numpy as np
import scipy as sp

from ..functions import matern_five_halves


def compute_pointing_matrix_ingredients(x_list, side_list, bilinear: bool | tuple[bool] = True):

    if isinstance(bilinear, bool):
        bilinear = len(x_list) * [bilinear]

    if (len(x_list) != len(side_list)) or (len(x_list) != len(bilinear)):
        raise ValueError("")

    samples_shape = np.broadcast_shapes(*[x.shape for x in x_list])
    x_list = [x.reshape(samples_shape) for x in x_list]

    samples = np.arange(x_list[0].size, dtype=int).reshape(samples_shape)
    pixels = np.zeros(x_list[0].shape, dtype=int)
    weights = np.ones(x_list[0].shape, dtype=float)
    n_pixels = 1

    for dim_index, (x, side, dim_is_bilinear) in enumerate(zip(x_list, side_list, bilinear)):
        if np.size(side) > 1:
            pixels *= len(side)
            n_pixels *= len(side)

            padded_side = np.array([-np.inf, *side, np.inf])

            if dim_is_bilinear:
                bin_index = np.digitize(x, bins=side)
                p = (x - padded_side[bin_index]) / np.diff(padded_side)[bin_index]
                p = np.where(p > 0, p, 0)
                dim_pixels = np.stack([bin_index - 1, bin_index], axis=0).clip(0, len(side) - 1)
                dim_weights = np.stack([1 - p, p], axis=0)

            else:
                bin_index = np.digitize(x, bins=0.5 * (side[1:] + side[:-1]))
                dim_pixels = bin_index[None]
                dim_weights = np.ones_like(x, dtype=float)[None]

            for add_dim in range(dim_index):
                dim_pixels = np.expand_dims(dim_pixels, add_dim + 1)
                dim_weights = np.expand_dims(dim_weights, add_dim + 1)

            samples = samples + np.zeros_like(dim_pixels)
            pixels = pixels + dim_pixels
            weights = weights * dim_weights

    return (
        samples.reshape(-1, *samples_shape),
        pixels.reshape(-1, *samples_shape),
        weights.reshape(-1, *samples_shape),
        n_pixels,
        x_list[0].size,
    )


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


def generate_spatial_basis(offsets, k: int = 5, n_side: int = 8, scale: float = 1):

    x = np.linspace(offsets[..., 0].min(), offsets[..., 0].max(), n_side)
    y = np.linspace(offsets[..., 1].min(), offsets[..., 1].max(), n_side)

    X, Y = np.meshgrid(x, y)

    sample_offsets = np.stack([X.ravel(), Y.ravel()], axis=-1)

    D_eff = np.sqrt(np.square(sample_offsets - sample_offsets[:, None]).sum(axis=-1)) / scale

    C = matern_five_halves(D_eff)

    u, s, v = np.linalg.svd(C)
    basis = u[:, :k] * np.sqrt(s[:k])
    B = sp.interpolate.RegularGridInterpolator(
        (x, y),
        basis.reshape(n_side, n_side, -1),
        method="cubic",
    )(offsets)
    B *= np.sign(B[:, 0].mean())
    return B
