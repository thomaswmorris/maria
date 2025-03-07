from __future__ import annotations

import logging
import pathlib
from collections.abc import Mapping

import astropy as ap
import h5py
import pandas as pd
import yaml

logger = logging.getLogger("maria")


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


def read_yaml(path: str):
    """
    Return a YAML file as a dict
    """
    res = yaml.safe_load(pathlib.Path(path).read_text())
    return res if res is not None else {}


def test_file(path) -> bool:
    ext = path.split(".")[-1]
    try:
        if ext in ["h5"]:
            with h5py.File(path, "r") as f:
                f.keys()
        elif ext in ["csv"]:
            pd.read_csv(path)
        elif ext in ["txt", "dat"]:
            with open(path) as f:
                f.read()
        elif ext in ["fits"]:
            ap.io.fits.open(path)
    except Exception:
        return False

    return True
