import numpy as np


def get_vapor_pressure(air_temp, rel_hum):  # units are (°K, %)
    T = air_temp - 273.15  # in °C
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    gamma = np.log(1e-2 * rel_hum) + b * T / (c + T)
    return a * np.exp(gamma)


def get_saturation_pressure(air_temp):  # units are (°K, %)
    T = air_temp - 273.15  # in °C
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    return a * np.exp(b * T / (c + T))


def get_dew_point(air_temp, rel_hum):  # units are (°K, %)
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    p_vap = get_vapor_pressure(air_temp, rel_hum)
    return c * np.log(p_vap / a) / (b - np.log(p_vap / a)) + 273.15


def get_relative_humidity(air_temp, dew_point):
    T, DP = air_temp - 273.15, dew_point - 273.15  # in °C
    b, c = 17.67, 238.88
    return 1e2 * np.exp(b * DP / (c + DP) - b * T / (c + T))


def relative_to_absolute_humidity(air_temp, rel_hum):
    return 1e-2 * rel_hum * get_saturation_pressure(air_temp) / (461.5 * air_temp)


def absolute_to_relative_humidity(air_temp, abs_hum):
    return 1e2 * 461.5 * air_temp * abs_hum / get_saturation_pressure(air_temp)
