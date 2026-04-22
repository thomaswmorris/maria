import numpy as np


def hav(x):
    return (1 - np.cos(x)) / 2


def great_circle_distance(phi1, theta1, phi2, theta2):
    hav_d = hav(theta2 - theta1) + np.cos(theta1) * np.cos(theta2) * hav(phi1 - phi2)
    return 2 * np.sqrt(hav_d)
