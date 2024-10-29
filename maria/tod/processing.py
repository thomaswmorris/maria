import logging

import numpy as np
import scipy as sp

from .. import utils
from .tod import TOD

logger = logging.getLogger("maria")

PROCESS_KWARGS = {
    "window": {
        "name": {"dtype": str, "aliases": ["window"]},
        "kwargs": {"dtype": dict, "aliases": ["window_kwargs"]},
    },
    "filter": {
        "f_lower": {"dtype": float, "aliases": ["filter_f_lower"]},
        "f_upper": {"dtype": float, "aliases": ["filter_f_upper"]},
        "order": {"dtype": int, "aliases": ["filter_order"]},
        "method": {"dtype": str, "aliases": ["filter_method"]},
    },
    "remove_modes": {
        "modes_to_remove": {"dtype": list, "aliases": []},
    },
    "despline": {
        "knot_spacing": {"dtype": float, "aliases": ["despline_knot_spacing"]},
        "order": {"dtype": int, "aliases": ["depline_order"]},
    },
}


def process_process_kwargs(**kwargs):  # lol
    config = {}

    for subprocess, subprocess_params in PROCESS_KWARGS.items():
        subconfig = {}

        for key, param in subprocess_params.items():
            for kwarg in list(kwargs.keys()):
                if kwarg in param["aliases"]:
                    subconfig[key] = param["dtype"](kwargs.pop(kwarg))
                    continue

        if subconfig:
            config[subprocess] = subconfig

    if len(kwargs) > 0:
        raise ValueError(f"Invalid kwargs for TOD processing: {kwargs}.")

    return config


def process_tod(tod, config=None, **kwargs):
    config = config or process_process_kwargs(**kwargs)

    D = tod.signal.compute()
    W = np.ones(D.shape)

    if "window" in config:
        window_function = getattr(sp.signal.windows, config["window"]["name"])
        W *= window_function(D.shape[-1], **config["window"].get("kwargs", {}))
        D = W * sp.signal.detrend(D, axis=-1)

    if "filter" in config:
        if "window" not in config:
            logger.warning("Filtering without windowing is not recommended.")

        if "f_upper" in config["filter"]:
            D = utils.signal.lowpass(
                D,
                fc=config["filter"]["f_upper"],
                sample_rate=tod.sample_rate,
                order=config["filter"].get("order", 1),
                method="bessel",
            )

        if "f_lower" in config["filter"]:
            D = utils.signal.highpass(
                D,
                fc=config["filter"]["f_lower"],
                sample_rate=tod.sample_rate,
                order=config["filter"].get("order", 1),
                method="bessel",
            )

    if "remove_modes" in config:
        U, V = utils.signal.decompose(
            D, downsample_rate=np.maximum(int(tod.sample_rate), 1), mode="uv"
        )
        U[:, config["remove_modes"]["modes_to_remove"]] = 0
        D = U @ V

    if "despline" in config:
        B = utils.signal.get_bspline_basis(
            tod.time,
            spacing=config["despline"]["knot_spacing"],
            order=config["despline"].get("order", 3),
        )

        A = np.linalg.inv(B @ B.T) @ B @ D.T
        D -= A.T @ B

    ptod = TOD(
        data={"total": {"data": D}},
        weight=W,
        coords=tod.coords,
        units=tod.units,
        dets=tod.dets,
        dtype=np.float32,
    )

    ptod.processing_config = config

    return ptod
