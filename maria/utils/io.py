import os
import pathlib
import time as ttime
from collections.abc import Mapping
from datetime import datetime

import pytz
import requests
import yaml


def flatten_config(m: str, prefix: str = ""):
    """
    Turn any dict into a mapping of mappings.
    """
    # if too shallow, add a dummy index
    if not all(isinstance(v, Mapping) for k, v in m.items()):
        return flatten_config({"": m}, prefix="")

    # recursion!
    items = []
    for k, v in m.items():
        new_key = f"{prefix}/{k}" if prefix else k
        if all(isinstance(vv, Mapping) for kk, vv in v.items()):
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


def cache_is_ok(path: str, max_cache_age: float = 86400):
    """
    Check if we need to reload the cache.
    """
    if not os.path.exists(path):
        return False

    cache_age = ttime.time() - os.path.getmtime(path)

    if cache_age > max_cache_age:
        return False

    return True


def fetch_cache(
    source_url: str,
    cache_path: str,
    max_cache_age: float = 7 * 86400,
    refresh: bool = False,
    chunk_size: int = 8192,
):
    """
    Download the cache if needed
    """
    cache_dir = os.path.dirname(cache_path)

    # make the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        print(f"created cache at {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

    if (not cache_is_ok(cache_path, max_cache_age=max_cache_age)) or refresh:
        print(f"updating cache from {source_url}")

        with requests.get(source_url, stream=True) as r:
            r.raise_for_status()
            with open(cache_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

        cache_size = os.path.getsize(cache_path)
        print(f"downloaded data ({1e-6 * cache_size:.01f} MB) to {cache_path}")


def datetime_handler(time):
    """
    Accepts any time format you can think of, spits out datetime object
    """
    if isinstance(time, (int, float)):
        return datetime.fromtimestamp(time).astimezone(pytz.utc)
    if isinstance(time, str):
        return datetime.fromisoformat(time).replace(tzinfo=pytz.utc)
