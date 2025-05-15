# for a given cross-section, extrude points orthogonally using a callback to compute covariance
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import scipy as sp
from pandas import DataFrame

logger = logging.getLogger("maria")


MIN_RES = {"2d": 5, "3d": 15}
MIN_RES_PER_BEAM = {"2d": 0.1, "3d": 0.5}
MIN_RES_PER_FOV = {"2d": 0.05, "3d": 0.1}

COV_MAT_JITTER = 1e-6


def generate_layers(
    sim,
    mode: str = "2d",
    angular: bool = True,
    max_height: float = 2e3,  # in meters
    min_res: float = None,  # in meters
    min_res_per_beam: float = None,
    min_res_per_fov: float = None,
    layer_spacing: float = 500,
) -> DataFrame:
    """
    Generate atmospheric layers.
    N.B.: We implicitly parametrize this model with a single, constant elevation.
    """

    min_res = min_res or MIN_RES[mode]
    min_res_per_beam = min_res_per_beam or MIN_RES_PER_BEAM[mode]
    min_res_per_fov = min_res_per_fov or MIN_RES_PER_FOV[mode]

    min_el = sim.boresight.el.min()

    h_samples = np.arange(0, max_height + 1e0, 1e-1)

    z_samples = h_samples / np.sin(min_el)

    fov = sim.instrument.dets.field_of_view.rad
    fwhm = sim.instrument.dets.one_detector_from_each_band().physical_fwhm(z_samples[:, None] + 1e-16).min(axis=1)
    r1 = min_res * np.ones(len(z_samples))
    r2 = min_res_per_beam * fwhm
    r3 = min_res_per_fov * z_samples * fov
    res_samples = np.minimum(1e3, np.maximum.reduce([r1, r2, r3]))

    def res_func(h):
        return sp.interpolate.interp1d(h_samples, res_samples)(h)

    if mode == "2d":
        h_boundaries = np.arange(0, max_height + layer_spacing, layer_spacing)
        process_index = np.arange(len(h_boundaries) - 1)
    elif mode == "3d":
        h_boundaries = [0]
        while True:
            new_h = h_boundaries[-1] + res_func(h_boundaries[-1])
            if new_h > max_height:
                break
            h_boundaries.append(h_boundaries[-1] + res_func(h_boundaries[-1]))
        h_boundaries = np.array(h_boundaries)
        process_index = 0

    h_centers = (h_boundaries[1:] + h_boundaries[:-1]) / 2

    weather = sim.atmosphere.weather

    weather_values = weather(altitude=sim.site.altitude + h_centers)

    layers = pd.DataFrame(weather_values)
    layers.insert(0, "process_index", process_index)
    layers.insert(1, "h", h_centers)
    layers.insert(2, "dh", np.diff(h_boundaries))
    layers.insert(3, "res", res_func(layers.h))
    layers.insert(4, "z", h_centers / np.sin(min_el))
    layers.insert(5, "angular", angular)

    h_boundaries = [0, *(layers.h.values[:-1] + layers.h.values[1:]) / 2, 1e5]

    for layer_index, (h1, h2) in enumerate(zip(h_boundaries[:-1], h_boundaries[1:])):
        dummy_h = sim.site.altitude + np.linspace(h1, h2, 1024)
        h = weather.altitude
        w = weather.absolute_humidity
        total_water = np.trapezoid(np.interp(dummy_h, h, w), x=dummy_h)
        layers.loc[layer_index, "total_water"] = total_water

    def boundary_layer_profile(h, h_0: float = 1e3, alpha: float = 1 / 7):
        return np.exp(-h / h_0) * h**alpha

    rel_var = boundary_layer_profile(layers.h.values) ** 2
    pwv_var = (weather.pwv * sim.atmosphere.pwv_rms_frac) ** 2 * rel_var / sum(rel_var)
    layers.loc[:, "pwv_rms"] = np.sqrt(pwv_var)

    if angular:
        layers.res /= layers.z

    return layers


def construct_extrusion_layers(
    points: float,
    res_func: Callable,
    z_min: float,
    z_max: float,
    mode: str = "3d",
    **mode_kwargs,
):
    triangulation = sp.spatial.Delaunay(points[..., 1:])
    layers = pd.DataFrame(columns=["x", "z", "n", "res", "indices"])

    layer_spacing = 500

    n = 0
    z = z_min if mode == "3d" else layer_spacing / 2

    while z < z_max:
        i = len(layers)

        res = res_func(z)

        # find a line
        wide_lp_x_dense = np.arange(points[..., 1].min(), points[..., 1].max(), 1e0)
        wide_lp_z_dense = z * np.ones(len(wide_lp_x_dense))
        wide_lp_dense = np.c_[wide_lp_x_dense, wide_lp_z_dense]
        interior = triangulation.find_simplex(wide_lp_dense) > -1
        lp_dense_x = wide_lp_x_dense[interior]
        n_lp = np.maximum(2, int(np.ptp(np.atleast_1d(lp_dense_x)) / res))
        lp_x = np.linspace(lp_dense_x.min() - 2 * res, lp_dense_x.max() + 2 * res, n_lp)

        layers.loc[i, "x"] = lp_x
        layers.loc[i, "z"] = z
        layers.loc[i, "n"] = n_lp
        layers.loc[i, "res"] = res
        layers.loc[i, "indices"] = n + np.arange(n_lp)

        z += res if mode == "3d" else layer_spacing
        n += n_lp

    cross_section_x = np.concatenate(layers.x.values)
    cross_section_z = np.concatenate(
        [entry.z * np.ones(entry.n) for index, entry in layers.iterrows()],
    )
    cross_section_points = np.concatenate(
        [cross_section_x[..., None], cross_section_z[..., None]],
        axis=-1,
    )

    extrusion_points = np.arange(
        points[..., 0].min() - 2 * layers.res.min(),
        points[..., 0].max() + 2 * layers.res.min(),
        layers.res.min(),
    )

    return layers, cross_section_points, extrusion_points
