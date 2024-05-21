import os

import astropy as ap
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..constants import k_B
from ..instrument import beams
from .map import Map

here, this_filename = os.path.split(__file__)


class MapMixin:
    """
    This simulates scanning over celestial sources.

    TODO: add errors
    """

    def _run(self, **kwargs):
        self.sample_maps()

    def _sample_maps(self):
        dx, dy = self.coords.offsets(frame=self.map.frame, center=self.map.center)

        self.data["map"] = 1e-16 * np.random.standard_normal(size=dx.shape)

        pbar = tqdm(self.instrument.bands, disable=not self.verbose)

        for band in pbar:
            pbar.set_description(f"Sampling map ({band.name})")

            band_mask = self.instrument.dets.band_name == band.name

            nu = np.linspace(band.nu_min, band.nu_max, 64)

            TRJ = sp.interpolate.interp1d(
                self.map.frequency,
                self.map.data,
                axis=0,
                kind="nearest",
                bounds_error=False,
                fill_value="extrapolate",
            )(nu)

            power_map = (
                1e12
                * k_B
                * np.trapz(band.passband(nu)[:, None, None] * TRJ, axis=0, x=1e9 * nu)
            )

            # nu is in GHz, f is in Hz
            nu_fwhm = beams.compute_angular_fwhm(
                fwhm_0=self.instrument.dets.primary_size.mean(),
                z=np.inf,
                f=1e9 * band.center,
            )
            nu_map_filter = beams.construct_beam_filter(fwhm=nu_fwhm, res=self.map.res)
            filtered_power_map = beams.separably_filter(power_map, nu_map_filter)

            self.data["map"][band_mask] += sp.interpolate.RegularGridInterpolator(
                (self.map.x_side, self.map.y_side),
                filtered_power_map,
                bounds_error=False,
                fill_value=0,
                method="linear",
            )((dx[band_mask], dy[band_mask]))

        # pbar = tqdm(enumerate(self.map.frequency), disable=not self.verbose)
        # for i, nu in pbar:
        #     pbar.set_description(f"Sampling input map ({nu} GHz)")

        #     # nu is in GHz, f is in Hz
        #     nu_fwhm = beams.compute_angular_fwhm(
        #         fwhm_0=self.instrument.dets.primary_size.mean(), z=np.inf, f=1e9 * nu
        #     )
        #     nu_map_filter = beams.construct_beam_filter(fwhm=nu_fwhm, res=self.map.res)
        #     filtered_nu_map_data = beams.separably_filter(
        #         self.map.data[i], nu_map_filter
        #     )

        #     # band_res_radians = 1.22 * (299792458 / (1e9 * nu)) / self.instrument.primary_size
        #     # band_res_pixels = band_res_radians / self.map.res
        #     # FWHM_TO_SIGMA = 2.355
        #     # band_beam_sigma_pixels = band_res_pixels / FWHM_TO_SIGMA

        #     # # band_beam_filter = self.instrument.

        #     # # filtered_map_data = sp.ndimage.convolve()
        #     det_freq_response = self.instrument.dets.passband(nu=np.array([nu]))[:, 0]
        #     det_mask = det_freq_response > -np.inf  # -1e-3

        #     samples = sp.interpolate.RegularGridInterpolator(
        #         (self.map.x_side, self.map.y_side),
        #         filtered_nu_map_data,
        #         bounds_error=False,
        #         fill_value=0,
        #         method="linear",
        #     )((dx[det_mask], dy[det_mask]))

        #     self.data["map"][det_mask] = self.instrument.dets.dP_dTRJ[:, None] * samples


def from_fits(
    filename: str,
    resolution: float,
    frequency: float,
    center: tuple,
    index: int = 0,
    units: str = "K_RJ",
    degrees: bool = True,
    frame: str = "az_el",
) -> Map:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    hudl = ap.io.fits.open(filename)

    frequency = np.atleast_1d(frequency)

    map_data = hudl[index].data
    if map_data.ndim < 2:
        raise ValueError("Map should have at least 2 dimensions.")
    elif map_data.ndim == 2:
        map_data = map_data[None]

    *_, map_n_y, map_n_x = map_data.shape

    map_width = resolution * map_n_x
    map_height = resolution * map_n_y

    # res_degrees = resolution if degrees else np.degrees(resolution)

    # if self.map_units == "Jy/pixel":
    #     for i, nu in enumerate(self.map_freqs):
    #         map_data[i] = map_data[i] / utils.units.KbrightToJyPix(
    #             1e9 * nu, res_degrees, res_degrees
    #         )

    return Map(
        data=map_data,
        header=hudl[0].header,
        frequency=frequency,
        width=np.radians(map_width) if degrees else map_width,
        height=np.radians(map_height) if degrees else map_height,
        center=np.radians(center) if degrees else center,
        degrees=False,
        frame=frame,
        # inbright=self.map_inbright,
        units=units,
    )
