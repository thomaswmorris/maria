import dask.array as da
import numpy as np


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
    BUFFER = da.random.random(size=((n_steps + n_i) * n_j))
    BUFFER[n_steps * n_j :] = values

    # remember that (e, c) -> n_c * e + c
    for buffer_index in np.arange(n_steps)[::-1]:
        BUFFER[buffer_index * n_j + np.arange(n_j)] = A @ BUFFER[
            n_j * (buffer_index + 1 + i_sample_index) + j_sample_index
        ] + B @ np.random.standard_normal(size=n_j)

    return BUFFER[: n_steps * n_j]


DEFAULT_ATMOSPHERE_CONFIG = {
    "min_depth": 500,
    "max_depth": 3000,
    "n_layers": 4,
    "min_beam_res": 4,
}


MIN_SAMPLES_PER_RIBBON = 2
JITTER_LEVEL = 1e-4
