import arrow


def get_utc_day_hour(t):
    a = arrow.get(t).to("utc")
    return a.hour + a.minute / 60 + a.second / 3600


def get_utc_year_day(t):
    return int(arrow.get(t).to("utc").format("DDD")) + get_utc_day_hour(t) / 24 - 1


def get_utc_year(t):
    return arrow.get(t).to("utc").year
