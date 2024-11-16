import logging
import os
import pathlib
import time
import time as ttime
from collections.abc import Mapping
from datetime import datetime

import astropy as ap
import h5py
import pandas as pd
import pytz
import requests
import yaml
from tqdm import tqdm

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
            with open(path, "r") as f:
                f.read()
        elif ext in ["fits"]:
            ap.io.fits.open(path)
    except Exception:
        return False

    return True


def cache_status(path: str, max_age: float = 30 * 86400, refresh: bool = False):
    """
    Check if we need to reload the cache.
    """
    if refresh:
        logger.debug(f"Forcing refresh of {path}.")
        return "force_refresh"

    if not os.path.exists(path):
        logger.debug(f"Cached file at {path} does not exist.")
        return "missing"

    if not test_file(path):
        logger.debug(f"Could not open cached file at {path}.")
        return "corrupted"

    cache_age = ttime.time() - os.path.getmtime(path)

    if cache_age > max_age:
        logger.debug(f"Cached file at {path} is too old.")
        return "old"

    return "ok"


def download_from_url(
    source_url: str,
    cache_path: str = None,
    chunk_size: int = 8192,
    verbose: bool = False,
):
    """
    Download the cache if needed.
    """
    cache_dir = os.path.dirname(cache_path)

    # make the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        logger.info(f"Downloading {source_url} to {cache_dir} …")

    os.makedirs(cache_dir, exist_ok=True)

    with requests.get(source_url, stream=True) as r:
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            chunks = tqdm(
                r.iter_content(chunk_size=chunk_size),
                desc=f"Updating cache from {source_url}",
                disable=not verbose,
            )
            for chunk in chunks:
                f.write(chunk)

    cache_size = os.path.getsize(cache_path)
    logger.info(f"Downloaded {1e-6 * cache_size:.01f} MB to {cache_path}.")

    if not test_file(cache_path):
        raise RuntimeError("Could not open cached file.")

    return cache_path


def fetch(
    source_path: str,
    cache_path: str = None,
    max_age: float = 7 * 86400,
    refresh: bool = False,
    url_base: str = "https://github.com/thomaswmorris/maria-data/raw/master",
    **download_kwargs,
):
    """
    Fetch a file from the repo.
    """
    cache_path = cache_path or f"/tmp/maria-data/{source_path}"
    url = f"{url_base}/{source_path}"

    status = cache_status(cache_path, max_age=max_age, refresh=refresh)

    if status != "ok":
        try:
            download_from_url(url, cache_path=cache_path, **download_kwargs)
        except Exception:
            if status == "old":
                logger.info(f"Could not download {url}, reverting to old cache.")
            else:
                raise RuntimeError(f"Could not download {url}.")
    return cache_path


def datetime_handler(time):
    """
    Accepts any time format you can think of, spits out datetime object
    """
    if isinstance(time, (int, float)):
        return datetime.fromtimestamp(time).astimezone(pytz.utc)
    if isinstance(time, str):
        return datetime.fromisoformat(time).replace(tzinfo=pytz.utc)
