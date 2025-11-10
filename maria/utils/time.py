import arrow
import numpy as np


def get_day_hour(t):
    a = arrow.get(t)
    return a.hour + a.minute / 60 + a.second / 3600 + int(a.format("SSS")) / 3600000


def get_utc_day_hour(t):
    return get_day_hour(arrow.get(t).to("utc"))


def get_utc_year_day(t, partial=True):
    whole_day = int(arrow.get(t).to("utc").format("DDD")) - 1
    if partial:
        return whole_day + get_utc_day_hour(t) / 24
    return whole_day


def get_utc_year(t):
    return arrow.get(t).to("utc").year


utc_day_hour = np.vectorize(get_utc_day_hour)
utc_year_day = np.vectorize(get_utc_year_day)
