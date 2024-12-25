import logging
import os
import requests

import time as ttime

from tqdm import tqdm

from ..utils.io import test_file

logger = logging.getLogger("maria")


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
    chunk_size: int = 2**12,
    max_retries: int = 4,
):
    """
    Download the cache if needed.
    """
    cache_dir = os.path.dirname(cache_path)

    # make the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        logger.debug(f"Creating {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

    status = {"fails": 0, "success": False}
    while not status["success"]:
        try:
            with requests.get(source_url, stream=True) as r:
                r.raise_for_status()
                total_size_bytes = int(r.headers.get("content-length", 0))
                # logger.info(f"Caching data from {source_url}")
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
            status["success"] = True

        except Exception as error:
            if status["fails"] < max_retries:
                status["fails"] += 1
            else:
                raise error

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
