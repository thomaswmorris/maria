# for a given cross-section, extrude points orthogonally using a callback to compute covariance
import logging
import time as ttime
from typing import Callable

import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm

from ..utils.linalg import fast_psd_inverse

logger = logging.getLogger("maria")


def construct_extrusion_layers(
    points: float,
    res_func: Callable,
    z_min: float,
    z_max: float,
    mode: str = "3d",
    **mode_kwargs,
):
    triangulation = sp.spatial.Delaunay(points[..., 1:])
    layers = pd.DataFrame(columns=["x", "z", "n", "res", "indices"])

    layer_spacing = 500

    n = 0
    z = z_min if mode == "3d" else layer_spacing / 2

    while z < z_max:
        i = len(layers)

        res = res_func(z)

        # find a line
        wide_lp_x_dense = np.arange(points[..., 1].min(), points[..., 1].max(), 1e0)
        wide_lp_z_dense = z * np.ones(len(wide_lp_x_dense))
        wide_lp_dense = np.c_[wide_lp_x_dense, wide_lp_z_dense]
        interior = triangulation.find_simplex(wide_lp_dense) > -1
        lp_dense_x = wide_lp_x_dense[interior]
        n_lp = np.maximum(2, int(np.ptp(np.atleast_1d(lp_dense_x)) / res))
        lp_x = np.linspace(lp_dense_x.min() - res, lp_dense_x.max() + res, n_lp)

        layers.loc[i, "x"] = lp_x
        layers.loc[i, "z"] = z
        layers.loc[i, "n"] = n_lp
        layers.loc[i, "res"] = res
        layers.loc[i, "indices"] = n + np.arange(n_lp)

        z += res if mode == "3d" else layer_spacing
        n += n_lp

    cross_section_x = np.concatenate(layers.x.values)
    cross_section_z = np.concatenate(
        [entry.z * np.ones(entry.n) for index, entry in layers.iterrows()]
    )
    cross_section_points = np.concatenate(
        [cross_section_x[..., None], cross_section_z[..., None]], axis=-1
    )

    extrusion_points = np.arange(
        points[..., 0].min() - 2 * layers.res.min(),
        points[..., 0].max() + 2 * layers.res.min(),
        layers.res.min(),
    )

    return layers, cross_section_points, extrusion_points


class ProcessExtrusion:
    def __init__(
        self,
        cross_section,
        extrusion,
        callback,
        callback_kwargs: dict = {},
        lookback_decay_rate: float = 2,
        jitter: float = 1e-6,
        MIN_SAMPLES_PER_LAYER: int = 4,
    ):
        self.cross_section = cross_section
        self.extrusion = extrusion
        self.callback = callback
        self.callback_kwargs = callback_kwargs
        self.lookback_decay_rate = lookback_decay_rate
        self.jitter = jitter
        self.MIN_SAMPLES_PER_LAYER = MIN_SAMPLES_PER_LAYER

        self.n_cross_section = len(self.cross_section)
        self.n_extrusion = len(self.extrusion)
        I, J = np.meshgrid(np.arange(self.n_cross_section), np.arange(self.n_extrusion))

        points = np.c_[self.extrusion[J][..., None], self.cross_section[I]]

        extrusion_indices = [
            0,
            *(2 ** np.arange(0, np.log(self.n_extrusion) / np.log(2))).astype(int),
            self.n_extrusion - 1,
        ]

        self.extrusion_sample_index = []
        self.cross_section_sample_index = []
        for i, extrusion_index in enumerate(extrusion_indices):
            n_ribbon_samples = np.minimum(
                np.maximum(
                    int(self.n_cross_section * (2) ** -(i)), self.MIN_SAMPLES_PER_LAYER
                ),
                self.n_cross_section,
            )
            # cross_section_indices = sorted(np.random.choice(a=self.n_cross_section, size=n_ribbon_samples, replace=False))
            cross_section_indices = np.unique(
                np.linspace(0, self.n_cross_section - 1, n_ribbon_samples).astype(int)
            )
            self.cross_section_sample_index.extend(cross_section_indices)
            self.extrusion_sample_index.extend(
                np.repeat(extrusion_index, len(cross_section_indices))
            )

        self.cross_section_sample_index = np.array(self.cross_section_sample_index)
        self.extrusion_sample_index = np.array(self.extrusion_sample_index)

        E_sample = points[
            self.extrusion_sample_index, self.cross_section_sample_index, 0
        ]
        X_sample = points[
            self.extrusion_sample_index, self.cross_section_sample_index, 1
        ]
        Y_sample = points[
            self.extrusion_sample_index, self.cross_section_sample_index, 2
        ]

        self.sample_points = np.c_[E_sample, X_sample, Y_sample]
        self.n_sample = len(self.sample_points)

        self.E_live_edge = points[0, :, 0] - np.gradient(self.extrusion).mean()
        self.X_live_edge = points[0, :, 1]
        self.Y_live_edge = points[0, :, 2]

        self.live_edge_points = np.c_[
            self.E_live_edge, self.X_live_edge, self.Y_live_edge
        ]
        self.n_live_edge = len(self.live_edge_points)

        n_side_warn = 1000

        if self.n_sample > n_side_warn:
            logger.warning(
                f"a large covariance matrix (n_side={self.n_sample}) will be generated; "
                f"inverting these matrices is very expensive."
            )

        self.points = da.from_array(points)

    def compute_covariance_matrices(self):
        # edge-edge upper {i,j}
        start_time = ttime.monotonic()
        i, j = np.triu_indices(self.n_live_edge, k=1)
        COV_E_E = np.eye(self.n_live_edge) + self.jitter
        COV_E_E[i, j] = self.callback(
            np.sqrt(
                np.square(self.live_edge_points[j] - self.live_edge_points[i]).sum(
                    axis=1
                )
            ),
            **self.callback_kwargs,
        )
        COV_E_E[j, i] = COV_E_E[i, j]
        logger.debug(
            f"computed edge-edge covariance {COV_E_E.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms."
        )

        # this one is explicit
        start_time = ttime.monotonic()
        COV_E_S = self.callback(
            np.sqrt(
                np.square(
                    self.sample_points[None] - self.live_edge_points[:, None]
                ).sum(axis=2)
            ),
            **self.callback_kwargs,
        )
        logger.debug(
            f"computed edge-sample covariance {COV_E_S.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms."
        )

        start_time = ttime.monotonic()
        # sample-sample upper {i,j}
        i, j = np.triu_indices(self.n_sample, k=1)
        COV_S_S = np.eye(self.n_sample) + self.jitter
        COV_S_S[i, j] = self.callback(
            np.sqrt(
                np.square(self.sample_points[j] - self.sample_points[i]).sum(axis=1)
            ),
            **self.callback_kwargs,
        )
        COV_S_S[j, i] = COV_S_S[i, j]
        logger.debug(
            f"computed sample-sample covariance {COV_S_S.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms."
        )

        # this is typically the bottleneck
        start_time = ttime.monotonic()
        inv_COV_S_S = fast_psd_inverse(COV_S_S)
        logger.debug(
            f"inverted sample-sample covariance {COV_S_S.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms."
        )

        self.COV_S_S = COV_S_S

        # compute the weights
        self.A = COV_E_S @ inv_COV_S_S

        if (self.A.sum(axis=-1) > 1.0).any():
            raise ValueError(
                f"propagation operator is unstable (A_max = {self.A.sum(axis=-1).max()})."
            )

        start_time = ttime.monotonic()
        self.B = np.linalg.cholesky(COV_E_E - self.A @ COV_E_S.T)

        duration_ms = 1e3 * (ttime.monotonic() - start_time)
        logger.debug(
            f"computed Cholesky decomposition of posterior covariance {COV_E_E.shape} in {duration_ms:.0f} ms."
        )

        self.values = np.zeros((self.n_extrusion, self.n_cross_section))

        initial_slice = self.B @ np.random.standard_normal(self.n_cross_section)
        initial_slice *= np.sqrt(np.diag(COV_E_E) / initial_slice.var())

        self.values[:] = initial_slice

    def run(self, desc="extruding"):
        if not hasattr(self, "A"):
            self.compute_covariance_matrices()

        n_steps = 2 * self.n_extrusion
        BUFFER = np.random.standard_normal(
            (self.n_extrusion + n_steps, self.n_cross_section)
        )
        for buffer_index in tqdm(np.arange(n_steps)[::-1], desc=desc):
            new_values = self.A @ BUFFER[
                buffer_index + self.extrusion_sample_index + 1,
                self.cross_section_sample_index,
            ] + self.B @ np.random.standard_normal(size=self.n_live_edge)
            BUFFER[buffer_index] = new_values

        self.values = da.from_array(BUFFER[: self.n_extrusion])
