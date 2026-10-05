import numpy as np

from ..constants import DRY_AIR_SPECIFIC_GAS_CONSTANT, WATER_VAPOR_SPECIFIC_GAS_CONSTANT, g

def vapor_pressure(temperature, humidity):  # units are (°K, %)
    T = temperature - 273.15  # in °C
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    gamma = np.log(np.maximum(humidity, 1e-20)) + b * T / (c + T)
    return a * np.exp(gamma)


def saturation_pressure(temperature):  # units are (°K, %)
    T = temperature - 273.15  # in °C
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    return a * np.exp(b * T / (c + T))


def dew_point(temperature, humidity):  # units are (°K, %)
    a, b, c = 611.21, 17.67, 238.88  # units are Pa, ., °C
    p_vap = vapor_pressure(temperature, humidity)
    return c * np.log(p_vap / a) / (b - np.log(p_vap / a)) + 273.15


def dew_point_to_relative_humidity(temperature, dew_point):
    T, DP = temperature - 273.15, dew_point - 273.15  # in °C
    b, c = 17.67, 238.88
    return 1e2 * np.exp(b * DP / (c + DP) - b * T / (c + T))


def compute_air_density(pressure, temperature, humidity):
    vp = vapor_pressure(temperature, humidity)
    return vp / (WATER_VAPOR_SPECIFIC_GAS_CONSTANT * temperature) + (pressure - vp) / (
        DRY_AIR_SPECIFIC_GAS_CONSTANT * temperature
    )


def relative_to_absolute_humidity(temperature, humidity):
    return humidity * saturation_pressure(temperature) / (461.5 * temperature)


def absolute_to_relative_humidity(temperature, abs_hum):
    return 461.5 * temperature * abs_hum / saturation_pressure(temperature)

