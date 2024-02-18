import os
from datetime import datetime

import h5py
import numpy as np
import scipy as sp
from tqdm import tqdm

from .. import utils
from ..sim.base import BaseSimulation
from ..site import InvalidRegionError, all_regions
from .turbulent_layer import TurbulentLayer
from .weather import Weather  # noqa F401

here, this_filename = os.path.split(__file__)

SPECTRA_DATA_DIRECTORY = f"{here}/data"
SPECTRA_DATA_CACHE_DIRECTORY = "/tmp/maria_data_cache/spectra"
SPECTRA_DATA_URL_BASE = (
    "https://github.com/thomaswmorris/maria-data/raw/master/spectra"  # noqa F401
)
MAX_CACHE_AGE_SECONDS = 30 * 86400


class Atmosphere:
    def __init__(
        self,
        t: float = None,
        region: str = "princeton",
        altitude: float = None,
        weather_quantiles: dict = {},
        weather_override: dict = {},
        weather_source: str = "era5",
        weather_from_cache: bool = None,
        spectrum_source: str = "am",
        spectrum_from_cache: bool = None,
    ):
        if region not in all_regions:
            raise InvalidRegionError(region)

        t = t or datetime.utcnow().timestamp()

        self.region = region
        self.spectrum_source = spectrum_source
        self.spectrum_source_path = (
            f"{SPECTRA_DATA_DIRECTORY}/{self.spectrum_source}/{self.region}.h5"
        )

        # if the data isn't in the module, default to use the cache
        self.spectrum_from_cache = (
            spectrum_from_cache
            if spectrum_from_cache is not None
            else not os.path.exists(self.spectrum_source_path)
        )

        if self.spectrum_from_cache:
            self.spectrum_source_path = f"{SPECTRA_DATA_CACHE_DIRECTORY}/{self.spectrum_source}/{self.region}.h5"
            utils.io.fetch_cache(
                source_url=f"{SPECTRA_DATA_URL_BASE}/{self.spectrum_source}/{self.region}.h5",
                cache_path=self.spectrum_source_path,
                max_cache_age=MAX_CACHE_AGE_SECONDS,
            )
            self.spectrum_from_cache = True

        with h5py.File(self.spectrum_source_path, "r") as f:
            self.spectrum_side_nu = f["side_nu_GHz"][:]
            self.spectrum_side_elevation = f["side_elevation_deg"][:]
            self.spectrum_side_zenith_pwv = f["side_zenith_pwv_mm"][:]
            self.spectrum_side_base_temperature = f["side_base_temperature_K"][:]

            self.emission_spectrum = f["emission_temperature_rayleigh_jeans_K"][:]
            self.transmission_spectrum = np.exp(-f["opacity_nepers"][:])
            self.excess_path_spectrum = 1e6 * (
                f["excess_path"][:] + f["offset_excess_path_m"][:]
            )

        self.weather = Weather(
            t=t,
            region=region,
            altitude=altitude,
            quantiles=weather_quantiles,
            override=weather_override,
            source=weather_source,
            from_cache=weather_from_cache,
        )

    def emission(self, nu):
        return

    def transmission(self, nu):
        return


class AtmosphereMixin:
    def _initialize_atmosphere(self):
        """
        This assume that BaseSimulation.__init__() has been called.
        """

        utils.validate_pointing(self.det_coords.az, self.det_coords.el)

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
                    self.instrument.physical_fwhm(z=layer_depth).min()
                    / self.min_atmosphere_beam_res
                )  # in meters

                layer = TurbulentLayer(
                    instrument=self.instrument,
                    boresight=self.boresight,
                    weather=self.atmosphere.weather,
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
            (self.n_atmosphere_layers, self.instrument.n_dets, self.pointing.n_time)
        )

        layers = tqdm(self.turbulent_layers) if self.verbose else self.turbulent_layers
        for layer_index, layer in enumerate(layers):
            if self.verbose:
                layers.set_description(f"Generating atmosphere (z={layer.depth:.00f}m)")

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
            self.atmosphere.weather.altitude_levels,
            self.atmosphere.weather.absolute_humidity,
        )
        rel_layer_scaling /= np.sqrt(np.square(rel_layer_scaling).sum(axis=0)[None])

        self.layer_scaling = (
            self.pwv_rms_frac * self.atmosphere.weather.pwv * rel_layer_scaling
        )

        self.zenith_scaled_pwv = self.atmosphere.weather.pwv + (
            self.layer_scaling * turbulence
        ).sum(axis=0)

    def _simulate_atmospheric_emission(self, units="K_RJ"):
        if units == "K_RJ":  # Kelvin Rayleigh-Jeans
            self._simulate_atmospheric_fluctuations()
            self.data["atmosphere"] = np.empty(
                (self.instrument.n_dets, self.pointing.n_time), dtype=np.float32
            )

            bands = (
                tqdm(self.instrument.dets.bands)
                if self.verbose
                else self.instrument.dets.bands
            )

            for band in bands:
                band_index = self.instrument.dets(band=band.name).uid

                if self.verbose:
                    bands.set_description(f"Computing atm. emission ({band.name})")

                # multiply by 1e9 to go from GHz to Hz
                # det_power_grid = (
                #     1e9
                #     * 1.380649e-23
                #     * np.trapz(
                #         self.atmosphere.emission_spectrum
                #         * band.passband(self.atmosphere.spectrum_side_nu),
                #         self.atmosphere.spectrum_side_nu,
                #         axis=-1,
                #     )
                # )

                # this is NOT power
                det_power_grid = np.sum(
                    self.atmosphere.emission_spectrum
                    * band.passband(self.atmosphere.spectrum_side_nu),
                    axis=-1,
                )
                det_power_grid /= np.sum(
                    band.passband(self.atmosphere.spectrum_side_nu), axis=-1
                )

                band_power_interpolator = sp.interpolate.RegularGridInterpolator(
                    (
                        self.atmosphere.spectrum_side_zenith_pwv,
                        self.atmosphere.spectrum_side_base_temperature,
                        self.atmosphere.spectrum_side_elevation,
                    ),
                    det_power_grid,
                )

                self.data["atmosphere"][band_index] = band_power_interpolator(
                    (
                        self.zenith_scaled_pwv[band_index],
                        self.atmosphere.weather.temperature[0],
                        np.degrees(self.det_coords.el[band_index]),
                    )
                )

            bands = (
                tqdm(self.instrument.dets.bands)
                if self.verbose
                else self.instrument.dets.bands
            )

            for band in bands:
                band_index = self.instrument.dets(band=band.name).uid

                if self.verbose:
                    bands.set_description(f"Computing atm. transmission ({band.name})")

                rel_T_RJ_spectrum = (
                    band.passband(self.atmosphere.spectrum_side_nu)
                    * self.atmosphere.spectrum_side_nu**2
                )

                # multiply by 1e9 to go from GHz to Hz
                self.det_transmission_grid = np.trapz(
                    rel_T_RJ_spectrum * self.atmosphere.transmission_spectrum,
                    self.atmosphere.spectrum_side_nu,
                    axis=-1,
                ) / np.trapz(
                    rel_T_RJ_spectrum,
                    self.atmosphere.spectrum_side_nu,
                    axis=-1,
                )

                band_transmission_interpolator = sp.interpolate.RegularGridInterpolator(
                    (
                        self.atmosphere.spectrum_side_zenith_pwv,
                        self.atmosphere.spectrum_side_base_temperature,
                        self.atmosphere.spectrum_side_elevation,
                    ),
                    self.det_transmission_grid,
                )

                self.atmospheric_transmission = band_transmission_interpolator(
                    (
                        self.zenith_scaled_pwv[band_index],
                        self.atmosphere.weather.temperature[0],
                        np.degrees(self.det_coords.el[band_index]),
                    )
                )

        if units == "F_RJ":  # Fahrenheit Rayleigh-Jeans 🇺🇸
            self._simulate_atmospheric_emission(self, units="K_RJ")
            self.data["atmosphere"] = 1.8 * (self.data["atmosphere"] - 273.15) + 32


class AtmosphereSimulation(BaseSimulation, AtmosphereMixin):
    def __init__(self, *args, **kwargs):
        super(AtmosphereSimulation, self).__init__(*args, **kwargs)
        self._initialize_atmosphere()
