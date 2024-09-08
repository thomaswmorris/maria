import logging
import os
from datetime import datetime

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..base import BaseSimulation
from ..coords import Coordinates, dx_dy_to_phi_theta
from ..functions import approximate_normalized_matern
from ..spectrum import AtmosphericSpectrum
from ..utils import compute_optimal_rotation
from ..weather import Weather  # noqa F401
from .extrusion import ProcessExtrusion, construct_extrusion_layers

here, this_filename = os.path.split(__file__)

# SPECTRA_DATA_DIRECTORY = f"{here}/data"
# SPECTRA_DATA_CACHE_DIRECTORY = "/tmp/maria-data/atmosphere/spectra"
# SPECTRA_DATA_URL_BASE = "https://github.com/thomaswmorris/maria-data/raw/master/atmosphere/spectra"  # noqa F401
# CACHE_MAX_AGE_SECONDS = 30 * 86400

logger = logging.getLogger("maria")


class Atmosphere:
    def __init__(
        self,
        model: str = "2d",
        timestamp: float = datetime.now().timestamp(),
        region: str = "princeton",
        altitude: float = None,
        weather_quantiles: dict = {},
        weather_kwargs: dict = {},
        weather_source: str = "era5",
        spectrum_source: str = "am",
        pwv_rms_frac: float = 0.03,
    ):
        self.timestamp = timestamp

        self.spectrum = AtmosphericSpectrum(
            region=region,
            source=spectrum_source,
        )

        self.weather = Weather(
            t=timestamp,
            region=region,
            altitude=altitude,
            quantiles=weather_quantiles,
            override=weather_kwargs,
            source=weather_source,
        )

        self.pwv_rms_frac = pwv_rms_frac

        self._initialized = False

        self.model = model

    def initialize(
        self, sim: BaseSimulation, timestep: float = 1e-1, z_min=1e1, z_max=5e3
    ):
        """
        Simulate a realization of PWV.
        """

        self.z_min = z_min
        self.z_max = z_max
        self.timestep = timestep
        self.sim = sim

        self.coords = self.sim.coords.downsample(timestep=self.timestep)

        self.wind_vector = np.array([10, 10, 0])

        fov_hull = sp.spatial.ConvexHull(sim.instrument.dets.offsets)
        fov_hull_offsets = fov_hull.points[fov_hull.vertices]

        self.boresight = sim.boresight.downsample(timestep=self.timestep)

        fov_hull_az, fov_hull_el = dx_dy_to_phi_theta(
            *fov_hull_offsets.T[..., None], self.boresight.az, self.boresight.el
        )

        c = Coordinates(
            time=self.boresight.time,
            phi=fov_hull_az,
            theta=fov_hull_el,
            earth_location=self.boresight.earth_location,
            frame="az_el",
        )

        logger.debug("constructed atmospheric coordinates")

        self.projected_fov_hull_points = np.concatenate(
            [
                (np.cos(c.az) / np.tan(c.el))[..., None],
                (np.sin(c.az) / np.tan(c.el))[..., None],
                (np.ones((*c.az.shape, 1))),
            ],
            axis=-1,
        )

        # add the origin, and transport the points with the wind vector
        points = (
            np.r_[
                z_max * self.projected_fov_hull_points,
                np.zeros((1, *self.projected_fov_hull_points.shape[1:])),
            ]
            + self.wind_vector * (c.time - c.time.min())[..., None]
        )
        total_hull = sp.spatial.ConvexHull(points.reshape(-1, 3))
        p = total_hull.points[total_hull.vertices]

        logger.debug("constructed atmospheric hull")

        self.transform = compute_optimal_rotation(p)
        tp = p @ self.transform  # transformed points

        v1 = sp.spatial.ConvexHull(p[:, 1:]).volume * np.ptp(p[:, 0])
        v2 = sp.spatial.ConvexHull(tp[:, 1:]).volume * np.ptp(tp[:, 0])

        logger.debug(
            f"optimized hull rotation (vol: {1e-9 * v1:.03f} km^3 -> Tvol: {1e-9 * v2:.03f} km^3"
        )

        MIN_RESOLUTION = 10
        MIN_RESOLUTION_PER_BEAM = 0.5
        MIN_RESOLUTION_PER_FOV = 0.05

        # construct a way to generate the layer resolution as a function of height
        z_samples = np.arange(z_min, z_max, 1e-1)
        fov = sim.instrument.dets.field_of_view.radians
        fwhm = sim.instrument.dets.angular_fwhm(z_samples[:, None]).min(axis=1)
        r1 = MIN_RESOLUTION * np.ones(len(z_samples))
        r2 = MIN_RESOLUTION_PER_BEAM * z_samples * fwhm
        r3 = MIN_RESOLUTION_PER_FOV * z_samples * fov
        res_samples = np.minimum(1e3, np.maximum.reduce([r1, r2, r3]))

        def res_func(z):
            return sp.interpolate.interp1d(z_samples, res_samples)(z)

        layers, cross_section_points, extrusion_points = construct_extrusion_layers(
            tp, res_func, z_min=z_min, z_max=z_max, mode=self.model
        )

        self.extrusion_res = np.gradient(extrusion_points).mean()
        self.cross_section_points = cross_section_points
        self.extrusion_points = extrusion_points
        self.layers = layers

        for field in [*self.weather.fields, "absolute_humidity"]:
            self.layers.loc[:, field] = np.interp(
                sim.site.altitude + self.layers.z,
                self.weather.altitude,
                getattr(self.weather, field),
            )

        matern_kwargs = (
            {"nu": 1 / 3, "r0": 3e2} if self.model == "3d" else {"nu": 5 / 6, "r0": 3e2}
        )

        self.turbulence_process = ProcessExtrusion(
            cross_section=cross_section_points,
            extrusion=extrusion_points,
            callback=approximate_normalized_matern,
            callback_kwargs=matern_kwargs,
            jitter=1e-8,
        )

        self.turbulence_process.compute_covariance_matrices()

        self._initialized = True

    def simulate_pwv(self):
        if not self._initialized:
            raise RuntimeError("Atmosphere must be initialized with a simulation.")

        self.turbulence_process.run(desc="Generating atmosphere")
        t = self.coords.time - self.coords.time.min()

        pp = np.concatenate(
            [
                (np.cos(self.coords.az) / np.tan(self.coords.el))[..., None],
                (np.sin(self.coords.az) / np.tan(self.coords.el))[..., None],
                (np.ones((*self.coords.az.shape, 1))),
            ],
            axis=-1,
        )

        self.zenith_scaled_pwv = da.from_array(np.zeros(pp.shape[:-1]))

        layer_boundaries = [
            0,
            *(self.layers.z.values[1:] + self.layers.z.values[:-1]) / 2,
            self.z_max,
        ]

        for index in tqdm(self.layers.index, desc="Sampling atmosphere"):
            layer_bottom, layer_top = (
                layer_boundaries[index],
                layer_boundaries[index + 1],
            )
            thickness = layer_top - layer_bottom

            entry = self.layers.loc[index]

            beam_fwhm = self.sim.instrument.dets.physical_fwhm(entry.z).mean()
            beam_sigma = beam_fwhm / 2.355
            extrusion_sigma = beam_sigma / self.extrusion_res
            x_sigma = beam_sigma / entry.res

            smoothed_layer_pwv = sp.ndimage.gaussian_filter(
                self.turbulence_process.values[:, entry.indices],
                sigma=(extrusion_sigma, x_sigma),
            )

            # layer-projected points
            lpp = entry.z * pp + self.wind_vector * t[:, None]
            transformed_lpp = lpp @ self.transform
            y = sp.interpolate.RegularGridInterpolator(
                (
                    self.turbulence_process.extrusion,
                    self.turbulence_process.cross_section[entry.indices][..., 0],
                ),
                smoothed_layer_pwv,
            )(transformed_lpp[..., :2])

            self.zenith_scaled_pwv += (
                thickness * entry.absolute_humidity * (1 + 1e-2 * y)
            )
