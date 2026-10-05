import logging
import os
import shutil
import time as ttime

import astropy as ap
import h5py
import numpy as np
import pandas as pd
import requests
from requests import HTTPError
from tqdm import tqdm

from .logging import DEFAULT_BAR_FORMAT

logger = logging.getLogger("maria")
here, this_filename = os.path.split(__file__)


def set_local_cache_dir(directory):
    os.environ["MARIA_LOCAL_CACHE_DIR"] = directory


def get_local_cache_dir():
    return os.environ.get("MARIA_LOCAL_CACHE_DIR", "/tmp/maria-data")

def set_data_repo(url):
    os.environ["MARIA_DATA_REPO"] = url

def get_data_repo():
    return os.environ.get("MARIA_DATA_REPO", "https://github.com/thomaswmorris/maria-data/raw/master")

def copy_file(source, destination):
    dest_dir, _ = os.path.split(destination)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(source, destination)


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
            logger.info(f"Fetching {source_url}")
            with tqdm(
                total=total_size_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Downloading",
                bar_format=DEFAULT_BAR_FORMAT,
                ncols=250,
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
    path: str = None,
    url: str = None,
    max_age: float = None,
    refresh: bool = False,
    max_attempts: int = 7,
    **download_kwargs,
):
    """
    Fetch a file from the repo.
    """

    max_age = max_age or float(os.environ.get("MARIA_CACHE_MAX_AGE", 30 * 86400))

    if path:
        url = f"{get_data_repo()}/{path}"
    elif url:
        path = f"misc/{os.path.split(url)[-1]}"
    else:
        raise ValueError("You must pass one of 'path' or 'url'.")


    local_cache_dir = get_local_cache_dir()
    local_cache_path = f"{local_cache_dir}/{path}"

    # if source_path:
    #     source_url = f"{url_base}/{source_path}"
    #     local_cache_path = local_cache_path or f"{local_cache_dir}/{source_path}"
    # elif source_url is not None:
    #     _, tail = os.path.split(source_url)
    #     local_cache_path = local_cache_path or f"{local_cache_dir}/{tail}"
    # else:
    #     raise RuntimeError("You must supply either 'source_url' or 'source_path'.")

    # do we need to do anything?
    status = cache_status(local_cache_path, max_age=max_age, refresh=refresh)

    if status == "ok":
        return local_cache_path

    # do we have a potential backup?
    if status == "stale":
        stale_cache_path = f"{local_cache_dir}/stale/{path}"
        copy_file(local_cache_path, stale_cache_path)
    else:
        stale_cache_path = None

    attempt = 0
    while attempt < max_attempts:
        status = download_from_url(url, cache_path=local_cache_path, max_age=max_age, **download_kwargs)
        if status == "ok":
            return local_cache_path
        attempt += 1
        logger.warning(f"Could not download {url} on try {attempt} (status = {status})")
        ttime.sleep(2e0)

    if stale_cache_path:
        logger.warning(f"Could not download {url}, using stale cache at {stale_cache_path}")
        return stale_cache_path

    raise RuntimeError(f"Could not download {url} after {max_attempts} retries (status = {status})")
