from __future__ import annotations

import os

import dask.array as da
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..constants import k_B

here, this_filename = os.path.split(__file__)


class AtmosphereMixin:
    def _simulate_atmosphere(self):
        # this produces self.atmosphere.zenith_scaled_pwv, which we use to compute emission and opacity
        self.atmosphere.simulate_pwv()

    def _compute_atmospheric_emission(self):
        self.atmosphere.emission = da.zeros_like(self.atmosphere.zenith_scaled_pwv)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Computing atmospheric emission",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_index = self.instrument.dets.mask(band_name=band.name)

            # in picowatts. the 1e9 is for GHz -> Hz
            det_power_grid = (
                1e12
                * k_B
                * np.trapezoid(
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

            self.atmosphere.emission[band_index] = band_power_interpolator(
                (
                    self.atmosphere.zenith_scaled_pwv[band_index],
                    self.atmosphere.weather.temperature[0],
                    np.degrees(self.atmosphere.coords.el[band_index]),
                ),
            )

        self.data["atmosphere"] = da.from_array(
            sp.interpolate.interp1d(
                self.atmosphere.coords.t,
                self.atmosphere.emission,
                bounds_error=False,
                fill_value="extrapolate",
            )(self.coords.t),
        )

    def _compute_atmospheric_opacity(self):
        self.atmosphere.opacity = da.zeros_like(self.atmosphere.zenith_scaled_pwv)

        bands_pbar = tqdm(
            self.instrument.dets.bands,
            desc="Computing atmospheric opacity",
            disable=self.disable_progress_bars,
        )
        for band in bands_pbar:
            bands_pbar.set_postfix({"band": band.name})

            band_index = self.instrument.dets.mask(band_name=band.name)

            _nu = self.atmosphere.spectrum._side_nu
            _tau = band.passband(_nu)

            band_opacity_grid = -np.log(
                np.trapezoid(
                    np.exp(-self.atmosphere.spectrum._opacity) * _tau,
                    x=1e9 * _nu,
                    axis=-1,
                )
                / np.trapezoid(band.passband(_nu), x=1e9 * _nu, axis=-1),
            )

            # self.det_opacity_grid = np.trapezoid(
            #     rel_T_RJ_spectrum * self.atmosphere.spectrum._opacity,
            #     1e9 * self.atmosphere.spectrum._side_nu,
            #     axis=-1,
            # ) / np.trapezoid(
            #     rel_T_RJ_spectrum,
            #     1e9 * self.atmosphere.spectrum._side_nu,
            #     axis=-1,
            # )

            band_opacity_interpolator = sp.interpolate.RegularGridInterpolator(
                (
                    self.atmosphere.spectrum._side_zenith_pwv,
                    self.atmosphere.spectrum._side_base_temperature,
                    self.atmosphere.spectrum._side_elevation,
                ),
                band_opacity_grid,
            )

            # what's happening here? the atmosphere blocks some of the light from space.
            # we want to calibrate to the stuff in space, so we make the atmosphere *hotter*

            self.atmosphere.opacity[band_index] = band_opacity_interpolator(
                (
                    self.atmosphere.zenith_scaled_pwv[band_index],
                    self.atmosphere.weather.temperature[0],
                    np.degrees(self.atmosphere.coords.el[band_index]),
                ),
            )

        self.atmospheric_transmission = np.exp(
            -da.from_array(
                sp.interpolate.interp1d(
                    self.atmosphere.coords.t,
                    self.atmosphere.opacity,
                    bounds_error=False,
                    fill_value="extrapolate",
                )(self.coords.t),
            ),
        )
