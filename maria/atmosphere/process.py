from __future__ import annotations

import logging
import time as ttime

import dask.array as da
import numpy as np
from tqdm import tqdm

from ..functions import approximate_normalized_matern
from ..io import humanize_time
from ..utils import fast_psd_inverse

logger = logging.getLogger("maria")

COV_MAT_JITTER = 1e-6


class AutoregressiveProcess:
    def __init__(
        self,
        cross_section,
        extrusion,
        callback=approximate_normalized_matern,
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
        I, J = np.meshgrid(np.arange(self.n_cross_section), np.arange(self.n_extrusion))  # noqa

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
                    int(self.n_cross_section * (2) ** -(i)),
                    self.MIN_SAMPLES_PER_LAYER,
                ),
                self.n_cross_section,
            )
            # cross_section_indices = sorted(np.random.choice(a=self.n_cross_section, size=n_ribbon_samples, replace=False))
            cross_section_indices = np.unique(
                np.linspace(0, self.n_cross_section - 1, n_ribbon_samples).astype(int),
            )
            self.cross_section_sample_index.extend(cross_section_indices)
            self.extrusion_sample_index.extend(
                np.repeat(extrusion_index, len(cross_section_indices)),
            )

        self.cross_section_sample_index = np.array(self.cross_section_sample_index)
        self.extrusion_sample_index = np.array(self.extrusion_sample_index)

        E_sample = points[
            self.extrusion_sample_index,
            self.cross_section_sample_index,
            0,
        ]
        X_sample = points[
            self.extrusion_sample_index,
            self.cross_section_sample_index,
            1,
        ]
        Y_sample = points[
            self.extrusion_sample_index,
            self.cross_section_sample_index,
            2,
        ]

        self.sample_points = np.c_[E_sample, X_sample, Y_sample]
        self.n_sample = len(self.sample_points)

        self.E_live_edge = points[0, :, 0] - np.gradient(self.extrusion).mean()
        self.X_live_edge = points[0, :, 1]
        self.Y_live_edge = points[0, :, 2]

        self.live_edge_points = np.c_[
            self.E_live_edge,
            self.X_live_edge,
            self.Y_live_edge,
        ]
        self.n_live_edge = len(self.live_edge_points)

        n_side_warn = 1000

        if self.n_sample > n_side_warn:
            logger.warning(
                f"A large covariance matrix (n_side={self.n_sample}) will be generated; inverting these matrices is very expensive.",  # noqa
            )

        self.points = da.asarray(points)

    def compute_covariance_matrices(self):
        # edge-edge upper {i,j}
        covariance_s = ttime.monotonic()
        i, j = np.triu_indices(self.n_live_edge, k=1)
        COV_E_E = np.eye(self.n_live_edge) + self.jitter
        COV_E_E[i, j] = self.callback(
            np.sqrt(
                np.square(self.live_edge_points[j] - self.live_edge_points[i]).sum(
                    axis=1,
                ),
            ),
            **self.callback_kwargs,
        )
        COV_E_E[j, i] = COV_E_E[i, j]
        COV_E_E += np.diag(COV_MAT_JITTER * np.diag(COV_E_E))  # add some jitter
        # logger.debug(
        #     f"Computed edge-edge covariance {COV_E_E.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms.",
        # )

        # this one is explicit
        COV_E_S = self.callback(
            np.sqrt(
                np.square(
                    self.sample_points[None] - self.live_edge_points[:, None],
                ).sum(axis=2),
            ),
            **self.callback_kwargs,
        )
        # logger.debug(
        #     f"Computed edge-sample covariance {COV_E_S.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms.",
        # )

        # sample-sample upper {i,j}
        i, j = np.triu_indices(self.n_sample, k=1)
        COV_S_S = np.eye(self.n_sample) + self.jitter
        COV_S_S[i, j] = self.callback(
            np.sqrt(
                np.square(self.sample_points[j] - self.sample_points[i]).sum(axis=1),
            ),
            **self.callback_kwargs,
        )
        COV_S_S[j, i] = COV_S_S[i, j]
        COV_S_S += np.diag(COV_MAT_JITTER * np.diag(COV_S_S))  # add some jitter
        # logger.debug(
        #     f"Computed sample-sample covariance {COV_S_S.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms.",
        # )

        # this is typically the bottleneck
        inv_COV_S_S = fast_psd_inverse(COV_S_S)
        # logger.debug(
        #     f"Inverted sample-sample covariance {COV_S_S.shape} in {1e3 * (ttime.monotonic() - start_time):.0f} ms.",
        # )

        self.COV_S_S = COV_S_S

        # compute the weights
        self.A = COV_E_S @ inv_COV_S_S

        if (self.A.sum(axis=-1) > 1.0).any():
            raise ValueError(
                f"Propagation operator is unstable (A_max = {self.A.sum(axis=-1).max()}).",
            )

        self.B = np.linalg.cholesky(COV_E_E - self.A @ COV_E_S.T)

        # logger.debug(
        #     f"Computed Cholesky decomposition {COV_E_E.shape} in {duration_ms:.0f} ms.",
        # )

        self.values = np.zeros((self.n_extrusion, self.n_cross_section))

        initial_slice = self.B @ np.random.standard_normal(self.n_cross_section)
        initial_slice *= np.sqrt(np.diag(COV_E_E) / initial_slice.var())

        self.values[:] = initial_slice

        logger.debug(
            f"Computed propagators with shape {self.A.shape} in {humanize_time(ttime.monotonic() - covariance_s)}",
        )

    def run(self):
        if not hasattr(self, "A"):
            self.compute_covariance_matrices()

        n_steps = 2 * self.n_extrusion
        BUFFER = np.random.standard_normal(
            (self.n_extrusion + n_steps, self.n_cross_section),
        )

        iterator = np.arange(n_steps)[::-1]

        for buffer_index in iterator:
            new_values = self.A @ BUFFER[
                buffer_index + self.extrusion_sample_index + 1,
                self.cross_section_sample_index,
            ] + self.B @ np.random.standard_normal(size=self.n_live_edge)
            BUFFER[buffer_index] = new_values

        self.values = da.asarray(BUFFER[: self.n_extrusion])
