import logging
import os
import shutil
import time as ttime

import numpy as np
import requests
from requests import HTTPError
from tqdm import tqdm

from ..utils.io import test_file

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)


def set_cache_dir(directory):
    os.environ["MARIA_CACHE_DIR"] = directory


def copy_file(source, destination):
    dest_dir, _ = os.path.split(destination)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(source, destination)


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
        logger.debug(f"Cached file at {path} is stale.")
        return "stale"

    return "ok"


def download_from_url(
    source_url: str,
    cache_path: str = None,
    chunk_size: int = 2**12,
    max_age: int = 30 * 86400,
):
    """
    Download the cache if needed.
    """
    cache_dir = os.path.dirname(cache_path)

    # make the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        logger.debug(f"Creating {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

    try:
        with requests.get(source_url, stream=True) as r:
            r.raise_for_status()
            total_size_bytes = int(r.headers.get("content-length", 0))
            with tqdm(
                total=total_size_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {source_url}",
            ) as pbar:
                with open(cache_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))

    except HTTPError as error:
        if error.response.status_code == 404:
            raise error
        return f"Encountered error while downloading {source_url}: {repr(error)}"

    return cache_status(cache_path, max_age=max_age, refresh=False)


def fetch(
    source_path: str,
    cache_path: str = None,
    max_age: float = 7 * 86400,
    refresh: bool = False,
    url_base: str = "https://github.com/thomaswmorris/maria-data/raw/master",
    max_attempts: int = 7,
    **download_kwargs,
):
    """
    Fetch a file from the repo.
    """

    cache_dir = os.environ.get("MARIA_CACHE_DIR", f"/tmp/maria-data")
    cache_path = cache_path or f"{cache_dir}/{source_path}"
    source_url = f"{url_base}/{source_path}"

    # do we need to do anything?
    status = cache_status(cache_path, max_age=max_age, refresh=refresh)

    if status == "ok":
        return cache_path

    # do we have a potential backup?
    if status == "stale":
        stale_cache_path = f"{cache_dir}/stale/{source_path}"
        copy_file(cache_path, stale_cache_path)
    else:
        stale_cache_path = None

    attempt = 0
    while attempt < max_attempts:
        status = download_from_url(source_url, cache_path=cache_path, max_age=max_age, **download_kwargs)
        if status == "ok":
            return cache_path
        attempt += 1
        logger.warning(f"Could not download {source_url} on try {attempt} (status = {status})")
        ttime.sleep(2e0)

    if stale_cache_path:
        logger.warning(f"Could not download {source_url}, using stale cache at {stale_cache_path}")
        return stale_cache_path

    raise RuntimeError(f"Could not download {source_url} after {max_attempts} retries (status = {status})")
