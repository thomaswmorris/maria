import pathlib
from datetime import datetime

import pytz
import yaml


def read_yaml(filepath):
    res = yaml.safe_load(pathlib.Path(filepath).read_text())
    return res if res is not None else {}


def datetime_handler(time):
    """
    Accepts any time format you can think of, spits out datetime object
    """
    if isinstance(time, (int, float)):
        return datetime.fromtimestamp(time).astimezone(pytz.utc)
    if isinstance(time, str):
        return datetime.fromisoformat(time).replace(tzinfo=pytz.utc)
