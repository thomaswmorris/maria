import os
import time as ttime

import h5py
import numpy as np
import scipy as sp
from tqdm import tqdm

from .. import base, utils, weather

# how do we do the bands? this is a great question.
# because all practical telescope instrumentation assume a constant band

here, this_filename = os.path.split(__file__)


def extrude(
    values: np.array,
    A: np.array,
    B: np.array,
    n_steps: int,
    n_i: int,
    n_j: int,
    i_sample_index: int,
    j_sample_index: int,
):
    # muy rapido
    BUFFER = np.zeros((n_steps + n_i) * n_j)
    BUFFER[n_steps * n_j :] = values

    # remember that (e, c) -> n_c * e + c
    for buffer_index in np.arange(n_steps)[::-1]:
        BUFFER[buffer_index * n_j + np.arange(n_j)] = A @ BUFFER[
            n_j * (buffer_index + 1 + i_sample_index) + j_sample_index
        ] + B @ np.random.standard_normal(size=n_j)

    return BUFFER[: n_steps * n_j]


class AtmosphericSpectrum:
    def __init__(self, filepath):
        """
        A dataclass to hold spectra as attributes
        """
        with h5py.File(filepath, "r") as f:
            self.side_nu_GHz = f["side_nu_GHz"][:].astype(float)
            self.side_elevation_deg = f["side_elevation_deg"][:].astype(float)
            self.side_line_of_sight_pwv_mm = f["side_line_of_sight_pwv_mm"][:].astype(
                float
            )
            self.temperature_rayleigh_jeans_K = f["temperature_rayleigh_jeans_K"][
                :
            ].astype(float)
            self.phase_delay_um = f["phase_delay_um"][:].astype(float)


class BaseAtmosphericSimulation(base.BaseSimulation):
    """
    The base class for modeling atmospheric fluctuations.

    The methods to simulate e.g. line-of-sight water and temeperature profiles should be implemented by
    classes which inherit from this one.
    """

    def __init__(self, array, pointing, site, verbose=False, **kwargs):
        super().__init__(array, pointing, site, verbose=verbose, **kwargs)

        utils.validate_pointing(self.pointing.az, self.pointing.el)

        self.weather = weather.Weather(
            t=self.pointing.time.mean(),
            region=self.site.region,
            altitude=self.site.altitude,
            quantiles=self.site.weather_quantiles,
        )

        spectrum_filepath = f"{here}/spectra/{self.site.region}.h5"
        self.spectrum = (
            AtmosphericSpectrum(filepath=spectrum_filepath)
            if os.path.exists(spectrum_filepath)
            else None
        )

    @property
    def EL(self):
        return utils.coords.xy_to_lonlat(
            self.array.offset_x[:, None],
            self.array.offset_y[:, None],
            self.pointing.az,
            self.pointing.el,
        )[1]

    def simulate_integrated_water_vapor(self):
        raise NotImplementedError(
            "Atmospheric simulations are not implemented in the base class!"
        )

    def _run(self, units="K_RJ"):
        start_time = ttime.monotonic()

        if units == "K_RJ":  # Kelvin Rayleigh-Jeans
            self.simulate_integrated_water_vapor()
            self.data = np.empty(
                (self.array.n_dets, self.pointing.n_time), dtype=np.float32
            )

            for uband in tqdm(self.array.ubands, desc="Sampling atmosphere"):
                # for uband in self.array.ubands:
                band_mask = self.array.dets.band.values == uband
                passband = self.array.passbands(self.spectrum.side_nu_GHz)[
                    band_mask
                ].mean(axis=0)

                band_T_RJ_interpolator = sp.interpolate.RegularGridInterpolator(
                    (
                        self.spectrum.side_line_of_sight_pwv_mm,
                        self.spectrum.side_elevation_deg,
                    ),
                    (self.spectrum.temperature_rayleigh_jeans_K * passband).sum(
                        axis=-1
                    ),
                )

                self.data[band_mask] = band_T_RJ_interpolator(
                    (
                        self.line_of_sight_pwv[band_mask],
                        np.degrees(self.EL[band_mask]),
                    )
                )

        if units == "F_RJ":  # Fahrenheit Rayleigh-Jeans 🇺🇸
            self.simulate_temperature(self, units="K_RJ")
            self.data = 1.8 * (self.data - 273.15) + 32

        if self.verbose:
            print(
                f"ran atmospheric simulation in {ttime.monotonic() - start_time:.01f} seconds"
            )


DEFAULT_ATMOSPHERE_CONFIG = {
    "min_depth": 500,
    "max_depth": 3000,
    "n_layers": 4,
    "min_beam_res": 4,
}


MIN_SAMPLES_PER_RIBBON = 2
JITTER_LEVEL = 1e-4
