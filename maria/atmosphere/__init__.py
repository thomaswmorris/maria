import os

import numpy as np
import scipy as sp
from tqdm import tqdm

from .. import utils
from ..base import BaseSimulation
from .spectra import AtmosphericSpectrum
from .turbulent_layer import TurbulentLayer
from .weather import Weather  # noqa F401

here, this_filename = os.path.split(__file__)


class AtmosphereMixin:
    def _initialize_atmosphere(self):
        """
        This assume that BaseSimulation.__init__() has been called.
        """

        utils.validate_pointing(self.det_coords.az, self.det_coords.el)

        self.spectrum = AtmosphericSpectrum(region=self.region)

        if self.atmosphere_model == "2d":
            self.turbulent_layer_depths = np.linspace(
                self.min_atmosphere_height,
                self.max_atmosphere_height,
                self.n_atmosphere_layers,
            )
            self.turbulent_layers = []

            depths = enumerate(self.turbulent_layer_depths)
            if self.verbose:
                depths = tqdm(depths, desc="Initializing atmospheric layers")

            for _, layer_depth in depths:
                layer_res = (
                    self.array.physical_fwhm(z=layer_depth).min()
                    / self.min_atmosphere_beam_res
                )  # in meters

                layer = TurbulentLayer(
                    array=self.array,
                    boresight=self.boresight,
                    weather=self.weather,
                    depth=layer_depth,
                    res=layer_res,
                    turbulent_outer_scale=self.turbulent_outer_scale,
                )

                self.turbulent_layers.append(layer)

        if self.atmosphere_model == "3d":
            self.initialize_3d_atmosphere()

    def _simulate_atmospheric_fluctuations(self):
        if self.atmosphere_model == "2d":
            self._simulate_2d_atmospheric_fluctuations()

        if self.atmosphere_model == "3d":
            self._simulate_3d_atmospheric_fluctuations()

    def _simulate_2d_turbulence(self):
        """
        Simulate layers of two-dimensional turbulence.
        """

        layer_data = np.zeros(
            (self.n_atmosphere_layers, self.array.n_dets, self.pointing.n_time)
        )

        layers = tqdm(self.turbulent_layers) if self.verbose else self.turbulent_layers
        for layer_index, layer in enumerate(layers):
            if self.verbose:
                layers.set_description(f"Generating atmosphere at {layer.depth:.00f}m")

            layer.generate()
            layer_data[layer_index] = layer.sample()

        return layer_data

    def _simulate_2d_atmospheric_fluctuations(self):
        """
        Simulate layers of two-dimensional turbulence.
        """

        turbulence = self._simulate_2d_turbulence()

        rel_layer_scaling = np.interp(
            self.site.altitude
            + self.turbulent_layer_depths[:, None, None] * np.sin(self.det_coords.el),
            self.weather.altitude_levels,
            self.weather.absolute_humidity,
        )
        rel_layer_scaling /= np.sqrt(np.square(rel_layer_scaling).sum(axis=0)[None])

        self.layer_scaling = self.pwv_rms_frac * self.weather.pwv * rel_layer_scaling

        self.effective_zenith_pwv = self.weather.pwv + (
            self.layer_scaling * turbulence
        ).sum(axis=0)

    def _simulate_atmospheric_emission(self, units="K_RJ"):
        if units == "K_RJ":  # Kelvin Rayleigh-Jeans
            self._simulate_atmospheric_fluctuations()
            self.data["atmosphere"] = np.empty(
                (self.array.n_dets, self.pointing.n_time), dtype=np.float32
            )

            band_names = self.array.dets.bands.keys()
            if self.verbose:
                band_names = tqdm(band_names)

            for band_name in band_names:
                band = self.array.dets.bands[band_name]

                if self.verbose:
                    band_names.set_description(
                        f"Sampling atmosphere for band {band_name}"
                    )

                band_mask = self.array.dets.band == band_name

                det_temperature_grid = np.trapz(
                    self.spectrum.emission * band.passband(self.spectrum.side_nu),
                    self.spectrum.side_nu,
                    axis=-1,
                )

                band_T_RJ_interpolator = sp.interpolate.RegularGridInterpolator(
                    (
                        self.spectrum.side_zenith_pwv,
                        self.spectrum.side_base_temperature,
                        self.spectrum.side_elevation,
                    ),
                    det_temperature_grid,
                )

                self.data["atmosphere"][band_mask] = band_T_RJ_interpolator(
                    (
                        self.effective_zenith_pwv[band_mask],
                        self.weather.temperature[0],
                        np.degrees(self.det_coords.el[band_mask]),
                    )
                )

        if units == "F_RJ":  # Fahrenheit Rayleigh-Jeans 🇺🇸
            self._simulate_atmospheric_emission(self, units="K_RJ")
            self.data["atmosphere"] = 1.8 * (self.data["atmosphere"] - 273.15) + 32


class AtmosphereSimulation(BaseSimulation, AtmosphereMixin):
    def __init__(self, *args, **kwargs):
        super(AtmosphereSimulation, self).__init__(*args, **kwargs)
        self._initialize_atmosphere()
