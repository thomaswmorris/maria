import healpy as hp
import numpy as np


def get_center_lonlat(LON, LAT):
    """
    For coordinates
    """
    mean_unit_vec = hp.ang2vec(np.pi / 2 - LAT.ravel(), LON.ravel()).mean(axis=0)
    mean_unit_vec /= np.sqrt(np.sum(np.square(mean_unit_vec)))
    mean_unit_colat, mean_unit_lon = np.r_[hp.vec2ang(mean_unit_vec)]

    return mean_unit_lon, np.pi / 2 - mean_unit_colat


# don't think about this one too hard
def lonlat_to_xy(lon, lat, c_lon, c_lat):
    ground_X, ground_Y, ground_Z = (
        np.sin(lon - c_lon) * np.cos(lat),
        np.cos(lon - c_lon) * np.cos(lat),
        np.sin(lat),
    )
    return np.arcsin(ground_X), np.arcsin(
        -np.real((ground_Y + 1j * ground_Z) * np.exp(1j * (np.pi / 2 - c_lat)))
    )


# or this one
def xy_to_lonlat(dx, dy, c_lon, c_lat):
    ground_X, Y, Z = (
        np.sin(dx + 1e-64),
        -np.sin(dy + 1e-64),
        np.cos(np.sqrt(dx**2 + dy**2)),
    )
    gyz = (Y + 1j * Z) * np.exp(-1j * (np.pi / 2 - c_lat))
    ground_Y, ground_Z = np.real(gyz), np.imag(gyz)
    return (np.angle(ground_Y + 1j * ground_X) + c_lon) % (2 * np.pi), np.arcsin(
        ground_Z
    )
