import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from .bands import Band
from .beams import compute_angular_fwhm

here, this_filename = os.path.split(__file__)

SUPPORTED_ARRAY_PACKINGS = ["hex", "square", "sunflower"]
SUPPORTED_ARRAY_SHAPES = ["hex", "square", "circle"]


def generate_2d_offsets(n, packing="hex", shape="circle", normalize=False):
    """
    Generate a scatter of $n$ points with some pattern.
    These points are spread such that each is a unit of distance away from its nearest neighbor.
    """

    n = int(n)
    bigger_n = 2 * n

    if packing == "square":
        s = int(np.ceil(np.sqrt(bigger_n)))
        side = np.arange(-s, s + 1, dtype=float)
        x, y = [foo.ravel() for foo in np.meshgrid(side, side)]

    elif packing == "hex":
        s = int(np.ceil((np.sqrt(12 * bigger_n - 3) + 3) / 6))
        side = np.arange(-s, s + 1, dtype=float)
        x, y = np.meshgrid(side, side)
        y[:, ::2] -= 0.5
        x *= np.sqrt(3) / 2
        x, y = x.ravel(), y.ravel()

    elif packing == "sunflower":
        i = np.arange(bigger_n)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        x = 0.5966 * np.sqrt(i) * np.cos(golden_angle * i)
        y = 0.5966 * np.sqrt(i) * np.sin(golden_angle * i)

    else:
        raise ValueError(
            "Supported offset packings are ['square', 'hex', or 'sunflower']."
        )

    n_sides = {"square": 4, "hex": 6, "circle": 256}[shape]

    r = np.sqrt(x**2 + y**2)
    p = np.arctan2(y, x)
    ngon_distance = r * np.cos(np.arcsin(np.sin(n_sides / 2 * p)) * 2 / n_sides)

    subset_index = np.argsort(ngon_distance)[:n]

    offsets = np.c_[x[subset_index], y[subset_index]]

    if normalize:
        hull_pts = offsets[ConvexHull(offsets).vertices]
        offsets /= cdist(hull_pts, hull_pts, metric="euclidean").max()

    return offsets


def generate_2d_offsets_from_diameter(
    diameter, packing="hex", shape="circle", tol=1e-2, max_iterations=32
):
    n = np.square(diameter)
    span = 0

    for i in range(max_iterations):
        offsets = generate_2d_offsets(n=n, packing=packing, shape=shape)
        ch = ConvexHull(offsets)
        hull_pts = ch.points[ch.vertices]
        span = cdist(hull_pts, hull_pts, metric="euclidean").max()

        n *= np.square(diameter / span)

        if np.abs(span - diameter) / diameter < tol:
            return offsets

    return offsets


def generate_array(
    bands: list,
    n: int = None,
    primary_size: float = 10.0,
    field_of_view: float = 0.0,
    beam_spacing: float = 1.0,
    array_packing: tuple = "hex",
    array_shape: tuple = "circle",
    array_offset: tuple = (0.0, 0.0),
    baseline_diameter: float = 0,
    baseline_packing: str = "sunflower",
    baseline_shape: str = "circle",
    baseline_offset: tuple = (0.0, 0.0, 0.0),
    polarization: bool = False,
    bath_temp: float = 0,
    file: str = None,
):
    dets = pd.DataFrame()

    bands = [Band.from_config(name=k, config=v) for k, v in bands.items()]

    band_centers = [band.center for band in bands]
    resolutions = [
        compute_angular_fwhm(primary_size, z=np.inf, f=1e9 * band.center)
        for band in bands
    ]
    detector_spacing = beam_spacing * np.max(resolutions)

    if n is not None:
        if field_of_view is not None:
            offsets = np.radians(field_of_view) * generate_2d_offsets(
                n=n, packing=array_packing, shape=array_shape, normalize=True
            )
        else:
            if len(resolutions) > 1:
                warnings.warn(
                    "Subarray has more than one band. "
                    f"Generating detector spacing based on the lowest frequency ({np.min(band_centers):.01f}) GHz."
                )
            offsets = detector_spacing * generate_2d_offsets(
                n=n, packing=array_packing, shape=array_shape
            )

    else:
        topological_diameter = np.radians(field_of_view) / detector_spacing
        offsets = detector_spacing * generate_2d_offsets_from_diameter(
            diameter=topological_diameter, packing=array_packing, shape=array_shape
        )

    baselines = baseline_diameter * generate_2d_offsets(
        n=len(offsets), packing=baseline_packing, shape=baseline_shape, normalize=True
    )

    if polarization:
        # generate random polarization angles and double each detector
        pol_angles = np.random.uniform(low=0, high=2 * np.pi, size=len(offsets))
        pol_labels = np.r_[["A" for _ in pol_angles], ["B" for _ in pol_angles]]
        pol_angles = np.r_[pol_angles, (pol_angles + np.pi / 2) % (2 * np.pi)]
        offsets = np.r_[offsets, offsets]
        baselines = np.r_[baselines, baselines]

    else:
        pol_angles = np.zeros(len(offsets))
        pol_labels = ["A" for i in pol_angles]

    for band in bands:
        band_dets = pd.DataFrame(
            index=np.arange(len(offsets)),
            dtype=float,
        )

        band_dets.loc[:, "band_name"] = band.name
        band_dets.loc[:, "sky_x"] = np.radians(array_offset[0]) + offsets[:, 0]
        band_dets.loc[:, "sky_y"] = np.radians(array_offset[1]) + offsets[:, 1]

        band_dets.loc[:, "baseline_x"] = baseline_offset[0] + baselines[:, 0]
        band_dets.loc[:, "baseline_y"] = baseline_offset[1] + baselines[:, 1]
        band_dets.loc[:, "baseline_z"] = baseline_offset[2]

        band_dets.loc[:, "bath_temp"] = bath_temp
        band_dets.loc[:, "pol_angle"] = pol_angles
        band_dets.loc[:, "pol_label"] = pol_labels

        band_dets.loc[:, "primary_size"] = primary_size

        dets = pd.concat([dets, band_dets])

    return dets
