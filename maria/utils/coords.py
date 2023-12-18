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


# def dx_dy_to_phi_theta(dx, dy, center_phi, center_theta):
#     """
#     Convert array offsets to e.g. az and el.
#     """
#     # Face north, and look up at the zenith. Here, the translation is
#     #
#     #

#     input_shape = np.shape(dx)
#     _dx, _dy = np.atleast_1d(dx).ravel(), np.atleast_1d(dy).ravel()

#     # if we're looking at phi=0, theta=pi/2, then we have:
#     phi = np.arctan2(_dx, -_dy)
#     theta = np.pi / 2 - np.sqrt(_dx**2 + _dy**2)

#     x = np.cos(phi) * np.cos(theta)
#     y = np.sin(phi) * np.cos(theta)
#     z = np.sin(theta)

#     points = np.c_[x, y, z].T
#     points = get_rotation_matrix_3d(angles=np.pi / 2 - center_theta, axis=1) @ points
#     points = get_rotation_matrix_3d(angles=-center_phi, axis=2) @ points

#     new_phi = np.arctan2(points[1], points[0]) % (2 * np.pi)
#     new_theta = np.arcsin(points[2])

#     return new_phi.reshape(input_shape), new_theta.reshape(input_shape)


# def phi_theta_to_dx_dy(phi, theta, center_phi, center_theta):
#     """
#     This is the inverse of the other one.
#     """
#     # Face north, and look up at the zenith. Here, the translation is
#     input_shape = np.shape(phi)
#     _phi, _theta = np.atleast_1d(phi).ravel(), np.atleast_1d(theta).ravel()

#     x = np.cos(_phi) * np.cos(_theta)
#     y = np.sin(_phi) * np.cos(_theta)
#     z = np.sin(_theta)

#     points = np.c_[x, y, z].T
#     points = get_rotation_matrix_3d(angles=center_phi, axis=2) @ points
#     points = get_rotation_matrix_3d(angles=center_theta - np.pi / 2, axis=1) @ points

#     p = np.angle(points[0] + 1j * points[1])
#     r = np.arccos(points[2])

#     return (r * np.sin(p)).reshape(input_shape), (-r * np.cos(p)).reshape(input_shape)


# don't think about this one too hard
def phi_theta_to_dx_dy(lon, lat, c_lon, c_lat):
    ground_X, ground_Y, ground_Z = (
        np.sin(lon - c_lon) * np.cos(lat),
        np.cos(lon - c_lon) * np.cos(lat),
        np.sin(lat),
    )
    return np.arcsin(ground_X), np.arcsin(
        -np.real((ground_Y + 1j * ground_Z) * np.exp(1j * (np.pi / 2 - c_lat)))
    )


# or this one
def dx_dy_to_phi_theta(dx, dy, c_lon, c_lat):
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
