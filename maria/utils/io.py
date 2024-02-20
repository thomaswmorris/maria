import os
import pathlib
import time as ttime
from collections.abc import Mapping
from datetime import datetime

import pytz
import requests
import yaml


def flatten_config(m, prefix=""):
    items = []
    for k, v in m.items():
        new_key = f"{prefix}/{k}" if prefix else k
        if any(isinstance(vv, Mapping) for kk, vv in v.items()):
            items.extend(flatten_config(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def read_yaml(path):
    """
    Return a YAML file as a dict
    """
    res = yaml.safe_load(pathlib.Path(path).read_text())
    return res if res is not None else {}


def fetch_cache(source_url, cache_path, max_cache_age=7 * 86400, refresh=False):
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
        r = requests.get(source_url)
        with open(cache_path, "wb") as f:
            f.write(r.content)

        cache_size = os.path.getsize(cache_path)
        print(f"downloaded data ({1e-6 * cache_size:.01f} MB) to {cache_path}")


def cache_is_ok(path, max_cache_age=86400):
    """
    Check if we need to reload the cache.
    """
    if not os.path.exists(path):
        return False

    cache_age = ttime.time() - os.path.getmtime(path)

    if cache_age > max_cache_age:
        return False

    return True


def datetime_handler(time):
    """
    Accepts any time format you can think of, spits out datetime object
    """
    if isinstance(time, (int, float)):
        return datetime.fromtimestamp(time).astimezone(pytz.utc)
    if isinstance(time, str):
        return datetime.fromisoformat(time).replace(tzinfo=pytz.utc)
