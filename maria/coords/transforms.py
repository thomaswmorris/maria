import numpy as np


def dx_dy_to_phi_theta(dx, dy, cphi, ctheta):
    """
    A fast and well-conditioned to convert from local dx/dy coordinates to phi/theta coordinates.
    """

    if not dx.shape == dy.shape:
        raise ValueError(
            f"The shapes of 'dx' and 'dy' must be the same. Got shapes {np.shape(dx)} and {np.shape(dy)}"
        )

    r = np.sqrt(dx**2 + dy**2)  # distance from the center
    p = np.arctan2(dx, -dy)  # 0 at the bottom, increases CCW to pi at the top

    # if we're looking at the north pole, we have (lon, lat) = (p, pi/2 - r)
    # a projection looking from the east
    proj_from_east = (np.sin(r) * np.cos(p) + 1j * np.cos(r)) * np.exp(
        1j * (ctheta - np.pi / 2)
    )
    phi = cphi + np.arctan2(np.sin(r) * np.sin(p), np.real(proj_from_east))
    theta = np.arcsin(np.imag(proj_from_east))

    return (
        phi,
        theta,
    )


def phi_theta_to_dx_dy(phi, theta, cphi, ctheta):
    """
    A fast and well-conditioned to convert from phi/theta coordinates to local dx/dy coordinates.
    """

    if not phi.shape == theta.shape:
        raise ValueError(
            f"The shapes of 'phi' and 'theta' must be the same. Got shapes {np.shape(phi)} and {np.shape(theta)}"
        )

    dphi = phi - cphi
    proj_from_east = (np.cos(dphi) * np.cos(theta) + 1j * np.sin(theta)) * np.exp(
        1j * (np.pi / 2 - ctheta)
    )
    dz = np.sin(dphi) * np.cos(theta) + 1j * np.real(proj_from_east)
    r = np.abs(dz)
    dz *= np.arcsin(r) / np.where(r > 0, r, 1.0)

    # negative, because we're looking at the observer
    return np.real(dz), -np.imag(dz)


def phi_theta_to_xyz(phi, theta):
    """
    Project a longitude and lattitude onto the unit sphere.
    """
    # you can add a newaxis on numpy floats, but not python floats. who knew?
    return np.concatenate(
        [
            (np.cos(phi) * np.cos(theta))[..., None],
            (np.sin(phi) * np.cos(theta))[..., None],
            (np.sin(theta))[..., None],
        ],
        axis=-1,
    )


def xyz_to_phi_theta(xyz):
    """
    Find the longitude and latitude of a 3-vector.
    """
    return np.arctan2(xyz[..., 1], xyz[..., 0]) % (2 * np.pi), np.arcsin(
        xyz[..., 2] / np.sqrt(np.sum(xyz**2, axis=-1))
    )


def get_center_phi_theta(phi, theta, keep_last_dim=False):
    """ """
    xyz = phi_theta_to_xyz(phi, theta)

    if keep_last_dim:
        center_xyz = xyz.mean(axis=tuple(range(xyz.ndim - 2)))
        center_xyz /= np.sqrt(np.sum(np.square(center_xyz), axis=-1))[..., None]
    else:
        center_xyz = xyz.mean(axis=tuple(range(xyz.ndim - 1)))
        center_xyz /= np.sqrt(np.sum(np.square(center_xyz)))

    return xyz_to_phi_theta(center_xyz)
