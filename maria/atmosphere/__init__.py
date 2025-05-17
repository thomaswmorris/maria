from __future__ import annotations

import logging
import os
import time as ttime

import arrow
import numpy as np
import scipy as sp
from jax import scipy as jsp
from numpy.linalg import LinAlgError
from tqdm import tqdm

from ..functions import approximate_normalized_matern
from ..io import humanize_time
from ..units import Quantity
from ..utils import compute_aligning_transform
from ..weather import Weather
from .extrusion import generate_layers
from .process import AutoregressiveProcess
from .spectrum import AtmosphericSpectrum

here, this_filename = os.path.split(__file__)

logger = logging.getLogger("maria")

SUPPORTED_MODELS_LIST = ["2d", "3d"]

DEFAULT_ATMOSPHERE_KWARGS = {}


class Atmosphere:
    def __init__(
        self,
        model: str = "2d",
        timestamp: float = arrow.now().timestamp(),
        region: str = "princeton",
        altitude: float = None,
        weather_quantiles: dict = {},
        weather_kwargs: dict = {},
        weather_source: str = "era5",
        spectrum_source: str = "am",
        pwv_rms_frac: float = 0.03,
        max_height: float = 5e3,
        timestep: float = 1e-1,
        disable_progress_bars: bool = False,
    ):
        if model not in SUPPORTED_MODELS_LIST:
            raise ValueError(
                f"Invalid model '{model}'. Supported models are {SUPPORTED_MODELS_LIST}.",
            )

        self.timestamp = timestamp

        self.spectrum = AtmosphericSpectrum(
            region=region,
            source=spectrum_source,
        )

        self.weather = Weather(
            time=timestamp,
            region=region,
            altitude=altitude,
            quantiles=weather_quantiles,
            override=weather_kwargs,
            source=weather_source,
        )

        self.pwv_rms_frac = pwv_rms_frac

        self.angular = False

        self.model = model
        self.max_height = max_height
        self.timestep = timestep

        self.disable_progress_bars = disable_progress_bars

        self._initialized = False

    def initialize(self, sim):
        """
        Simulate a realization of PWV.
        """

        self.sim = sim

        self.layers = generate_layers(
            sim,
            mode=self.model,
            angular=self.angular,
            max_height=self.max_height,
        )

        self.boresight = sim.boresight.downsample(timestep=self.timestep)
        self.coords = self.boresight.broadcast(
            sim.instrument.dets.offsets,
            frame="az_el",
        )

        # this is a smaller version of the sim coords
        outer_coords = self.boresight.broadcast(
            sim.instrument.dets.outer().offsets,
            frame="az_el",
        )

        self.processes = {}

        center = sim.coords.center(frame="az_el")

        for process_index in tqdm(
            sorted(np.unique(self.layers.process_index)),
            desc="Constructing atmosphere",
            disable=self.disable_progress_bars,
        ):
            process_init_s = ttime.monotonic()

            in_process = self.layers.process_index == process_index
            process_layers = self.layers.loc[in_process]

            # all processes must have the same velocity! so we compute it here.
            # vx and vy must have shape (n_time,)

            if process_layers.angular.any():
                vx = (
                    process_layers.wind_east.values[:, None] * np.cos(self.boresight.az)
                    - process_layers.wind_north.values[:, None] * np.sin(self.boresight.az)
                ) / process_layers.h.values[:, None]
                vy = (
                    -process_layers.wind_north.values[:, None] * np.cos(self.boresight.az)
                    - process_layers.wind_east.values[:, None] * np.sin(self.boresight.az)
                ) / (process_layers.h.values[:, None] * np.sin(self.boresight.el))
            else:
                vx = process_layers.wind_east.values[:, None] * np.ones(
                    self.boresight.shape[-1],
                )
                vy = process_layers.wind_north.values[:, None] * np.ones(
                    self.boresight.shape[-1],
                )

            w = (process_layers.absolute_humidity * process_layers.temperature * process_layers.divergence).values
            vx = (w[:, None] * vx).sum(axis=0) / w.sum()
            vy = (w[:, None] * vy).sum(axis=0) / w.sum()
            vz = np.zeros(vx.shape)

            if len(process_layers) > 1:
                process_layers = process_layers.loc[process_layers.index[[0, -1]]]

            # here we want enough points for a hull
            # we have made sure that any hull for this subpointing is a cover for the whole pointing

            process_points_for_hull_list = []

            for i, (layer_index, layer) in enumerate(process_layers.iterrows()):
                if layer.angular:
                    layer_x, layer_y = outer_coords.offsets(
                        center=center,
                        frame="az_el",
                    )

                else:
                    p = outer_coords.project(z=layer.h)
                    p += np.cumsum(self.timestep * np.c_[vx, vy, vz][None], axis=-2)

                    process_points_for_hull_list.append(p[None])

                    layer_x = layer.z * np.sin(outer_coords.az) * np.cos(outer_coords.el)
                    layer_y = layer.z * np.cos(outer_coords.az) * np.cos(outer_coords.el)

                layer_x += np.cumsum(self.timestep * vx)
                layer_y += np.cumsum(self.timestep * vy)

            process_points_for_hull = np.concatenate(
                process_points_for_hull_list,
                axis=0,
            ).reshape(-1, 3)
            process_points_for_hull[..., 2] += 1e-6 * np.random.standard_normal(
                process_points_for_hull[..., 2].shape,
            )
            # process_points_for_hull = process_points_for_hull.reshape(-1, 3)

            process_init_s = ttime.monotonic()

            transform = compute_aligning_transform(
                process_points_for_hull,
                signature=(True, True, False),
            )
            tp = process_points_for_hull @ transform
            triangulation = sp.spatial.Delaunay(tp[..., 1:])

            logger.debug(f"Computed process bounds in {humanize_time(ttime.monotonic() - process_init_s)}.")

            min_tx, min_ty, min_tz = tp.min(axis=0)
            max_tx, max_ty, max_tz = tp.max(axis=0)

            cross_section_points_list = []
            layer_labels = []

            for i, (layer_index, layer) in enumerate(
                self.layers.loc[in_process].iterrows(),
            ):
                res = layer.res
                wide_lp_x_dense = np.arange(min_ty - 2 * res, max_ty + 2 * res, 1e-1)
                wide_lp_z_dense = layer.h * np.ones(len(wide_lp_x_dense))
                wide_lp_dense = np.c_[wide_lp_x_dense, wide_lp_z_dense]
                interior = triangulation.find_simplex(wide_lp_dense) > -1
                lp_dense_x = wide_lp_x_dense[interior]
                lp_x_min = lp_dense_x.min() - 2 * res
                lp_x_max = lp_dense_x.max() + 2 * res
                n_lp = np.maximum(3, int((lp_x_max - lp_x_min) / res))
                lp_x = np.linspace(lp_x_min, lp_x_max, n_lp)
                lp_z = layer.h * np.ones(len(lp_x))
                lp = np.c_[lp_x, lp_z]
                n_lp = len(lp)

                layer_labels.extend(n_lp * [layer_index])
                cross_section_points_list.append(lp)

            cross_section_points = np.concatenate(cross_section_points_list, axis=0)

            extrusion_points = np.arange(
                min_tx - 2 * self.layers.res.min(),
                max_tx + 2 * self.layers.res.min(),
                self.layers.res.min(),
            )

            outer_scale = np.maximum(1e3, 300 + process_layers.h.mean() / 10)

            matern_kwargs = {"nu": 1 / 3, "r0": outer_scale} if self.model == "3d" else {"nu": 5 / 6, "r0": outer_scale}

            process = AutoregressiveProcess(
                cross_section=cross_section_points,
                extrusion=extrusion_points,
                callback=approximate_normalized_matern,
                callback_kwargs=matern_kwargs,
                jitter=1e-8,
            )

            process.layers = self.layers.loc[in_process]
            process.labels = np.array(layer_labels)
            process.transform = transform
            process.extrusion_res = np.gradient(extrusion_points).mean()
            process.vx = vx
            process.vy = vy
            process.tp = tp

            success = False
            for jitter in [1e-8, 1e-6, 1e-4]:
                try:
                    process.compute_covariance_matrices()
                    success = True
                    break
                except LinAlgError as e:
                    logger.debug(f"Singular covariance matrix with jitter={jitter}")

            if not success:
                raise LinAlgError("Covariance matrix was singular even with max jitter.")

            self.processes[int(process_index)] = process

        self._initialized = True

    def __repr__(self):
        return f"""Atmosphere({len(self.processes)} processes with {len(self.layers)} layers):
├ spectrum:
│   region: {self.spectrum.region}
└ weather:
    region: {self.weather.region}
    altitude: {Quantity(self.weather.base_altitude, "m")}
    time: {self.weather.local_time.format("MMM D HH:mm:ss ZZ")}
    pwv[mean, rms]: ({Quantity(self.weather.pwv, "mm")}, {Quantity(self.pwv_rms_frac * self.weather.pwv, "mm")})"""

    def simulate_pwv(self):
        if not self._initialized:
            raise RuntimeError("Atmosphere must be initialized with a simulation.")

        pp = self.coords.project(z=1)

        for k, process in tqdm(
            self.processes.items(),
            desc="Generating turbulence",
            disable=self.disable_progress_bars,
        ):
            process_s = ttime.monotonic()
            process.run()
            logger.debug(f"Ran atmospheric process {k} in {humanize_time(ttime.monotonic() - process_s)}")

        self.zenith_scaled_pwv = self.weather.pwv * np.ones(shape=pp.shape[:-1])

        with tqdm(
            total=len(self.layers),
            desc="Sampling turbulence",
            disable=self.disable_progress_bars,
        ) as pbar:
            for k, process in self.processes.items():
                wind_vector = np.c_[process.vx, process.vy, np.zeros(process.vx.shape)]
                translation = np.cumsum(self.timestep * wind_vector[None], axis=-2)

                for i, (layer_index, layer) in enumerate(
                    process.layers.iterrows(),
                ):
                    layer_s = ttime.monotonic()

                    layer_mask = process.labels == layer_index

                    beam_fwhm = self.sim.instrument.dets.physical_fwhm(
                        layer.z,
                    ).mean()
                    beam_sigma = beam_fwhm / 2.355
                    extrusion_sigma = beam_sigma / process.extrusion_res
                    x_sigma = beam_sigma / layer.res

                    smoothed_layer_pwv = sp.ndimage.gaussian_filter(
                        process.values[:, layer_mask],
                        sigma=(extrusion_sigma, x_sigma),
                    )

                    p = layer.h * pp + translation
                    transformed_lpp = p @ process.transform

                    y = jsp.interpolate.RegularGridInterpolator(
                        (
                            process.extrusion,
                            process.cross_section[layer_mask, 0],
                        ),
                        smoothed_layer_pwv,
                    )(transformed_lpp[..., :2])

                    self.zenith_scaled_pwv += layer.pwv_rms * y

                    pbar.update(1)

                    logger.debug(
                        f"Sampled layer {layer_index} (h={Quantity(layer.h, 'm')}) with shape "
                        f"{smoothed_layer_pwv.shape} in {humanize_time(ttime.monotonic() - layer_s)}"
                    )
