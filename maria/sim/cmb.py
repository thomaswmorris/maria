import dask.array as da
import healpy as hp
import numpy as np
import scipy as sp
from tqdm import tqdm

from ..functions import planck_spectrum
from ..units.constants import T_CMB


class CMBMixin:
    def _simulate_cmb_emission(self):
        pixel_index = hp.ang2pix(
            nside=self.cmb.nside, phi=self.coords.l, theta=np.pi / 2 - self.coords.b
        ).compute()  # noqa
        cmb_temperatures = self.cmb.T[pixel_index]

        test_nu = np.linspace(1e0, 1e3, 1024)

        cmb_temperature_samples_K = T_CMB + np.linspace(
            self.cmb.T.min(), self.cmb.T.max(), 3
        )  # noqa
        cmb_brightness = planck_spectrum(
            1e9 * test_nu, cmb_temperature_samples_K[:, None]
        )

        self.data["cmb"] = da.zeros_like(
            np.empty((self.instrument.dets.n, self.plan.n_time))
        )

        pbar = tqdm(self.instrument.bands, disable=not self.verbose)

        for band in pbar:
            pbar.set_description(f"Sampling CMB ({band.name})")

            band_mask = self.instrument.dets.band_name == band.name

            band_cmb_power_samples_W = (
                1e12
                * band.efficiency
                * np.trapezoid(y=cmb_brightness * band.passband(test_nu), x=test_nu)
            )

            # dP_dTCMB = self.instrument.dets.dP_dTCMB[:, None]
            # self.data["cmb"][band_mask] =  * cmb_temperatures[band_mask]

            self.data["cmb"][band_mask] = sp.interpolate.interp1d(
                cmb_temperature_samples_K,
                band_cmb_power_samples_W,
            )(T_CMB + cmb_temperatures[band_mask])
