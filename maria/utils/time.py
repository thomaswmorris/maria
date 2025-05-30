import arrow
import numpy as np

@np.vectorize
def utc_day_hour(t):
    a = arrow.get(t).to("utc")
    return a.hour + a.minute / 60 + a.second / 3600

@np.vectorize
def utc_year_day(t):
    return int(arrow.get(t).to("utc").format("DDD")) + utc_day_hour(t) / 24 - 1

def get_utc_year(t):
    return arrow.get(t).to("utc").year
