from typing import Mapping

import numpy as np

from ..units import Quantity
from ..utils import is_integer, is_numeric


def parse_t(t):
    """
    Infer from 't' the values of t in seconds
    """
    t_s_values = []
    for t_value in np.atleast_1d(t):
        if isinstance(t, Quantity):
            if not t.physical_quantity == "time":
                raise ValueError(f"'t' has units of {t_value.units} which are incompatible with time")
            t_s_values.append(t_value.to("s"))
        elif is_numeric(t_value):
            t_s_values.append(t_value)
        else:
            raise ValueError(
                "'t' must be either an array of floats (assumed to be a UNIX epoch) or a Quantity with dimensions of time"
            )

    return np.array(t_s_values, dtype=float)


def parse_nu(nu):
    """
    Infer from 'nu' the values of nu in Hz
    """

    nu_Hz_values = []
    for nu_value in np.atleast_1d(nu):
        if isinstance(nu_value, Quantity):
            if not nu_value.physical_quantity == "frequency":
                raise ValueError(f"'v' has units of {nu_value.units} which are incompatible with frequency")
            nu_Hz_values.append(nu_value.to("Hz"))
        elif is_numeric(nu_value):
            nu_Hz_values.append(nu_value)
        else:
            raise ValueError(
                "'nu' must be either an array of floats (assumed to be in units of Hz) "
                "or a Quantity with dimensions of frequency"
            )

    return np.array(nu_Hz_values, dtype=float)


def parse_v(v):
    """
    Infer from 'v' the values of v in Hz
    """

    v_mps_values = []
    for v_value in np.atleast_1d(v):
        if isinstance(v_value, Quantity):
            if not v_value.physical_quantity == "velocity":
                raise ValueError(f"'v' has units of {v_value.units} which are incompatible with frequency")
            v_mps_values.append(v_value.to("m/s"))
        elif is_numeric(v_value):
            v_mps_values.append(v_value)
        else:
            raise ValueError(
                "'v' must be either an array of floats (assumed to be in units of m/s) "
                "or a Quantity with dimensions of velocity"
            )

    return np.array(v_mps_values, dtype=float)


def parse_stokes(stokes):

    stokes_list = []

    if isinstance(stokes, str):
        stokes = list(stokes)

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

    return np.array(stokes_list)


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
