from __future__ import annotations

import numpy as np

from ..units import Quantity
from .coordinates import Coordinates, frames  # noqa
from .frame import FRAMES, Frame  # noqa
from .transforms import (  # noqa
    get_center_phi_theta,
    offsets_to_phi_theta,
    phi_theta_to_offsets,
    phi_theta_to_xyz,
    unjitted_offsets_to_phi_theta,
    xyz_to_phi_theta,
)


def infer_center_width_height(coords_list: list, frame: str = "ra_dec", center: tuple[Quantity] = None, square: bool = True):
    frame = Frame(frame)

    if center is None:
        center_phis, center_thetas = [], []
        for coords in coords_list:
            coords_center = coords.center(frame=frame)
            center_phis.append(coords_center[0].rad)
            center_thetas.append(coords_center[1].rad)

        center = get_center_phi_theta(center_phis, center_thetas)

    hull_points_list = []
    for coords in coords_list:
        hull_points_list.append(coords.hull(center=center, frame=frame))

    hull_points = np.concat(hull_points_list, axis=0)

    xmin, ymin = hull_points.min(axis=0)
    xmax, ymax = hull_points.max(axis=0)

    # corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
    # pt = unjitted_offsets_to_phi_theta(corners, *approx_center)
    # total_center = get_center_phi_theta(pt[..., 0], pt[..., 1])

    width, height = 2 * max(-xmin, xmax), 2 * max(-ymin, ymax)

    if square:
        width = height = max(width, height)

    return center, width, height
