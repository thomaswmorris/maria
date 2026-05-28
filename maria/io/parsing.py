from typing import Mapping

import numpy as np

from ..utils import is_integer


def parse_stokes(stokes):

    stokes_list = []
    for s in np.atleast_1d(stokes):
        if isinstance(s, str):
            if s in "IQUV":
                stokes_list.append(s)
            else:
                stokes_list = None
                break
        elif is_integer(s):
            try:
                stokes_list.append("IQUV"[int(s)])
            except Exception:
                stokes_list = None
                break

    if stokes_list is None:
        raise ValueError(
            f"Invalid Stokes parameters '{stokes}' (must be an iterable of parameters "
            "in ['I', 'Q', 'U', 'V'] or [0, 1, 2, 3])"
        )

    return "".join(stokes_list).upper()


def flatten_config(m: dict, prefix: str = ""):
    """
    Turn any dict into a mapping of mappings.
    """

    # if too shallow, add a dummy index
    if not isinstance(m, Mapping):
        return flatten_config({"": m}, prefix="")

    # if too shallow, add a dummy index
    if not all(isinstance(v, Mapping) for v in m.values()):
        return flatten_config({"": m}, prefix="")

    # recursion!
    items = []
    for k, v in m.items():
        new_key = f"{prefix}/{k}" if prefix else k
        if all(isinstance(vv, Mapping) for vv in v.values()):
            items.extend(flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
