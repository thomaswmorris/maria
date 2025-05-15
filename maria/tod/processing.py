from __future__ import annotations

import logging
import time as ttime

import numpy as np
import scipy as sp

from .. import utils
from ..io import humanize_time
from ..utils import remove_slope
from .tod import TOD

logger = logging.getLogger("maria")

OPERATION_KWARGS = {
    "remove_slope": {},
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
    "remove_spline": {
        "knot_spacing": {"dtype": float, "aliases": ["remove_spline_knot_spacing"]},
        "remove_el_gradient": {"dtype": bool, "aliases": ["remove_el_gradient"]},
        "order": {"dtype": int, "aliases": ["depline_order"]},
    },
}


def process_operation_kwargs(**kwargs):  # lol
    config = {}

    for operation, operation_params in OPERATION_KWARGS.items():
        if operation not in OPERATION_KWARGS:
            raise ValueError(f'Invalid operation "{operation}". Valid operations are {OPERATION_KWARGS.keys()}.')

        subconfig = {}
        for key, param in operation_params.items():
            for kwarg in list(kwargs.keys()):
                if kwarg in param["aliases"]:
                    subconfig[key] = kwargs.pop(kwarg)
                    continue

        if subconfig:
            config[operation] = subconfig

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

    if "remove_slope" in config:
        remove_slope_start_s = ttime.monotonic()
        D -= np.linspace(D[..., 0], D[..., -1], D.shape[-1]).T

        logger.debug(f'Completed tod operation "remove_slope in {humanize_time(ttime.monotonic() - remove_slope_start_s)}.')
        if np.isnan(D).any():
            raise ValueError("tod operation 'remove_slope' introduced NaNs")

    if "remove_spline" in config:
        remove_spline_start_s = ttime.monotonic()

        B = utils.signal.bspline_basis(
            tod.time,
            spacing=config["remove_spline"]["knot_spacing"],
            order=config["remove_spline"].get("order", 3),
        )

        if config["remove_spline"].get("remove_el_gradient", False):
            el = tod.boresight.el
            el_ptp = el.max() - el.min()
            if el_ptp == 0:
                raise ValueError("Cannot remove elevation gradient when elevation is constant")
            rel_el = (el - el.min()) / el_ptp
            B = np.concatenate([B * rel_el[None], B * (1 - rel_el)[None]], axis=0)

        A = np.linalg.inv(B @ B.T) @ B @ D.T
        D -= A.T @ B

        logger.debug(
            f'Completed tod operation "remove_spline" in {humanize_time(ttime.monotonic() - remove_spline_start_s)}.'
        )
        if np.isnan(D).any():
            raise ValueError("tod operation 'remove_spline' introduced NaNs")

    if "window" in config:
        window_start_s = ttime.monotonic()
        window_function = getattr(sp.signal.windows, config["window"]["name"])
        W *= window_function(D.shape[-1], **config["window"].get("kwargs", {}))
        D *= W
        logger.debug(f'Completed tod operation "window" in {humanize_time(ttime.monotonic() - window_start_s)}.')
        if np.isnan(D).any():
            raise ValueError("tod operation 'window' introduced NaNs")

    if "filter" in config:
        filter_start_s = ttime.monotonic()

        D = remove_slope(D)

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

        logger.debug(f'Completed tod operation "filter" in {humanize_time(ttime.monotonic() - filter_start_s)}.')
        if np.isnan(D).any():
            raise ValueError("tod operation 'filter' introduced NaNs")

    if "remove_modes" in config:
        remove_modes_start_s = ttime.monotonic()

        modes_to_remove = config["remove_modes"]["modes_to_remove"]
        A, B = utils.signal.decompose(D, k=np.max(modes_to_remove) + 1)
        D -= A[:, modes_to_remove] @ B[modes_to_remove]

        logger.debug(f'Completed tod operation "remove_modes" in {humanize_time(ttime.monotonic() - remove_modes_start_s)}.')
        if np.isnan(D).any():
            raise ValueError("tod operation 'remove_modes' introduced NaNs")

    ptod = TOD(
        data={"total": D},
        weight=W,
        coords=tod.coords,
        units=tod.units,
        dets=tod.dets,
        dtype=np.float32,
        metadata=tod.metadata,
    )

    ptod.processing_config = config

    return ptod
