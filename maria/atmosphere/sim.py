import os

import numpy as np
import scipy as sp
from tqdm import tqdm

from ..constants import k_B
from ..plan import validate_pointing
from .turbulent_layer import TurbulentLayer

here, this_filename = os.path.split(__file__)


class AtmosphereMixin:
    def _initialize_2d_atmosphere(
        self,
        min_atmosphere_height=500,
        max_atmosphere_height=5000,
        n_atmosphere_layers=4,
        min_atmosphere_beam_res=4,
        turbulent_outer_scale=500,
    ):
        """
        This assume that BaseSimulation.__init__() has been called.
        """

        validate_pointing(self.coords.az, self.coords.el)

        self.atmosphere_model = "2d"

        self.turbulent_layer_depths = np.linspace(
            min_atmosphere_height,
            max_atmosphere_height,
            n_atmosphere_layers,
        )
        self.atmosphere.layers = []

        depths = tqdm(
            self.turbulent_layer_depths,
            desc="Initializing atmospheric layers",
            disable=not self.verbose,
        )

        for layer_depth in depths:
            layer_res = (
                self.instrument.physical_fwhm(z=layer_depth).min()
                / min_atmosphere_beam_res
            )  # in meters

            layer = TurbulentLayer(
                instrument=self.instrument,
                boresight=self.boresight,
                weather=self.atmosphere.weather,
                depth=layer_depth,
                res=layer_res,
                turbulent_outer_scale=turbulent_outer_scale,
            )

            self.atmosphere.layers.append(layer)

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
            (len(self.atmosphere.layers), self.instrument.n_dets, self.plan.n_time)
        )

        pbar = tqdm(
            enumerate(self.atmosphere.layers),
            desc="Generating atmosphere",
            disable=not self.verbose,
        )

        for layer_index, layer in pbar:
            layer.generate()
            layer_data[layer_index] = sp.interpolate.interp1d(
                layer.sim_time,
                layer.sample(),
                axis=-1,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )(self.boresight.time)
            # pbar.set_description(f"Generating atmosphere (z={layer.depth:.00f}m)")

        return layer_data

    def _simulate_2d_atmospheric_fluctuations(self):
        """
        Simulate layers of two-dimensional turbulence.
        """

        turbulence = self._simulate_2d_turbulence()

        rel_layer_scaling = sp.interpolate.interp1d(
            self.atmosphere.weather.altitude_levels,
            self.atmosphere.weather.absolute_humidity,
            kind="linear",
        )(
            self.site.altitude
            + self.turbulent_layer_depths[:, None, None] * np.sin(self.coords.el)
        )
        rel_layer_scaling /= np.sqrt(np.square(rel_layer_scaling).sum(axis=0)[None])

        self.layer_scaling = (
            self.atmosphere.pwv_rms_frac
            * self.atmosphere.weather.pwv
            * rel_layer_scaling
        )

        self.zenith_scaled_pwv = self.atmosphere.weather.pwv + (
            self.layer_scaling * turbulence
        ).sum(axis=0)

    def _simulate_atmospheric_emission(self, units="K_RJ"):
        if units == "K_RJ":  # Kelvin Rayleigh-Jeans
            self._simulate_atmospheric_fluctuations()
            self.data["atmosphere"] = np.empty(
                (self.instrument.n_dets, self.plan.n_time), dtype=np.float32
            )

            bands = (
                tqdm(self.instrument.dets.bands)
                if self.verbose
                else self.instrument.dets.bands
            )

            for band in bands:
                band_index = self.instrument.dets.subset(band_name=band.name).index

                if self.verbose:
                    bands.set_description(f"Computing atm. emission ({band.name})")

                # in picowatts. the 1e9 is for GHz -> Hz
                det_power_grid = (
                    1e12
                    * k_B
                    * np.trapz(
                        self.atmosphere.spectrum._emission
                        * band.passband(self.atmosphere.spectrum._side_nu),
                        1e9 * self.atmosphere.spectrum._side_nu,
                        axis=-1,
                    )
                )

                band_power_interpolator = sp.interpolate.RegularGridInterpolator(
                    (
                        self.atmosphere.spectrum._side_zenith_pwv,
                        self.atmosphere.spectrum._side_base_temperature,
                        self.atmosphere.spectrum._side_elevation,
                    ),
                    det_power_grid,
                )

                self.data["atmosphere"][band_index] = band_power_interpolator(
                    (
                        self.zenith_scaled_pwv[band_index],
                        self.atmosphere.weather.temperature[0],
                        np.degrees(self.coords.el[band_index]),
                    )
                )

            self.atmospheric_transmission = np.empty(
                (self.instrument.n_dets, self.plan.n_time), dtype=np.float32
            )

            # to make a new progress bar
            bands = (
                tqdm(self.instrument.dets.bands)
                if self.verbose
                else self.instrument.dets.bands
            )

            for band in bands:
                band_index = self.instrument.dets.subset(band_name=band.name).index

                if self.verbose:
                    bands.set_description(f"Computing atm. transmission ({band.name})")

                rel_T_RJ_spectrum = (
                    band.passband(self.atmosphere.spectrum._side_nu)
                    * self.atmosphere.spectrum._side_nu**2
                )

                self.det_transmission_grid = np.trapz(
                    rel_T_RJ_spectrum * self.atmosphere.spectrum._transmission,
                    1e9 * self.atmosphere.spectrum._side_nu,
                    axis=-1,
                ) / np.trapz(
                    rel_T_RJ_spectrum,
                    1e9 * self.atmosphere.spectrum._side_nu,
                    axis=-1,
                )

                band_transmission_interpolator = sp.interpolate.RegularGridInterpolator(
                    (
                        self.atmosphere.spectrum._side_zenith_pwv,
                        self.atmosphere.spectrum._side_base_temperature,
                        self.atmosphere.spectrum._side_elevation,
                    ),
                    self.det_transmission_grid,
                )

                # what's happening here? the atmosphere blocks some of the light from space.
                # we want to calibrate to the stuff in space, so we make the atmosphere *hotter*

                self.atmospheric_transmission[
                    band_index
                ] = band_transmission_interpolator(
                    (
                        self.zenith_scaled_pwv[band_index],
                        self.atmosphere.weather.temperature[0],
                        np.degrees(self.coords.el[band_index]),
                    )
                )

        if units == "F_RJ":  # Fahrenheit Rayleigh-Jeans 🇺🇸
            self._simulate_atmospheric_emission(self, units="K_RJ")
            self.data["atmosphere"] = 1.8 * (self.data["atmosphere"] - 273.15) + 32
