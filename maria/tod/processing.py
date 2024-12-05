from __future__ import annotations

import logging

import numpy as np
import scipy as sp

from .. import utils
from .tod import TOD

logger = logging.getLogger("maria")

OPERATION_KWARGS = {
    "window": {
        "name": {"dtype": str, "aliases": ["window"]},
        "kwargs": {"dtype": dict, "aliases": ["window_kwargs"]},
    },
    "filter": {
        "f_lower": {"dtype": float, "aliases": ["f_lower"]},
        "f_upper": {"dtype": float, "aliases": ["f_upper"]},
        "order": {"dtype": int, "aliases": ["filter_order"]},
        "method": {"dtype": str, "aliases": ["filter_method"]},
    },
    "remove_modes": {
        "modes_to_remove": {"dtype": list, "aliases": ["modes_to_remove"]},
    },
    "despline": {
        "knot_spacing": {"dtype": float, "aliases": ["despline_knot_spacing"]},
        "order": {"dtype": int, "aliases": ["depline_order"]},
    },
}


def process_operation_kwargs(**kwargs):  # lol
    config = {}

    for subprocess, subprocess_params in OPERATION_KWARGS.items():
        subconfig = {}

        for key, param in subprocess_params.items():
            for kwarg in list(kwargs.keys()):
                if kwarg in param["aliases"]:
                    subconfig[key] = kwargs.pop(kwarg)
                    continue

        if subconfig:
            config[subprocess] = subconfig

    if len(kwargs) > 0:
        raise ValueError(f"Invalid kwargs for TOD processing: {kwargs}.")

    return config


def validate_process_config(config):
    for operation, operation_params in config.items():
        if operation not in OPERATION_KWARGS:
            raise ValueError(
                f"Invalid operation '{operation}'. Valid operations are {list(OPERATION_KWARGS.keys())}",
            )

        for key, value in operation_params.items():
            if key not in OPERATION_KWARGS[operation]:
                raise ValueError(
                    f"Invalid param '{key}' for operation '{operation}'. "
                    f"Valid parameters for this operation are {list(OPERATION_KWARGS[operation].keys())}",
                )

            dtype = OPERATION_KWARGS[operation][key]["dtype"]

            if not isinstance(value, dtype):
                try:
                    config[operation][key] = dtype(value)
                except Exception:
                    param = {key: value}
                    raise TypeError(
                        f"Could not convert param {param} for operation '{operation}' to requisite type '{dtype.__name__}'.",
                    )

    return config


def process_tod(tod, config=None, **kwargs):
    config = validate_process_config(config or process_operation_kwargs(**kwargs))

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

        modes_to_remove = config["remove_modes"]["modes_to_remove"]
        A, B = utils.signal.decompose(D, k=np.max(modes_to_remove) + 1)
        D -= A[:, modes_to_remove] @ B[modes_to_remove]

    if "despline" in config:
        B = utils.signal.get_bspline_basis(
            tod.time,
            spacing=config["despline"]["knot_spacing"],
            order=config["despline"].get("order", 3),
        )

        A = np.linalg.inv(B @ B.T) @ B @ D.T
        D -= A.T @ B

    ptod = TOD(
        data={"total": D},
        weight=W,
        coords=tod.coords,
        units=tod.units,
        dets=tod.dets,
        dtype=np.float32,
    )

    ptod.processing_config = config

    return ptod
