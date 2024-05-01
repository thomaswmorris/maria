import healpy as hp
import numpy as np
from tqdm import tqdm

from maria.constants import T_CMB
from maria.utils.functions import planck_spectrum

from ..utils.io import fetch_cache

PLANCK_URL = """https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/
                component-maps/cmb/COM_CMB_IQU-143-fgsub-sevem_2048_R3.00_full.fits"""


class CMB:
    def __init__(self, source="planck", nu=143):
        planck_url = PLANCK_URL
        cache_path = "/tmp/maria/cmb/planck/" + planck_url.split("/")[-1]

        fetch_cache(source_url=planck_url, cache_path=cache_path)

        field_dtypes = {
            "I": np.float32,
            "Q": np.float32,
            "U": np.float32,
            "I_mask": bool,
            "P_mask": bool,
        }

        for i, (field, dtype) in tqdm(
            enumerate(field_dtypes.items()), desc="Loading CMB"
        ):
            setattr(
                self, field, hp.fitsfunc.read_map(cache_path, field=i).astype(dtype)
            )

        self.source = source
        self.nside = 2048
        self.nu = nu

    def plot(self):
        vmin, vmax = 1e6 * np.quantile(self.I[self.I_mask], q=[0.001, 0.999])
        hp.visufunc.mollview(
            1e6 * np.where(self.I_mask, self.I, np.nan), min=vmin, max=vmax, cmap="cmb"
        )


class CMBMixin:
    def _initialize_cmb(self, source="planck", nu=150):
        self.cmb = CMB(source=source, nu=nu)

    def _simulate_cmb_emission(self):
        pixel_index = hp.ang2pix(
            nside=self.cmb.nside, phi=self.coords.l, theta=np.pi / 2 - self.coords.b
        ).compute()
        cmb_temperatures = self.cmb.I[pixel_index]

        test_nu = np.linspace(1e9, 1e12, 1024)

        cmb_temperature_samples_K = T_CMB + np.linspace(
            self.cmb.I.min(), self.cmb.I.max(), 64
        )
        cmb_brightness = planck_spectrum(test_nu, cmb_temperature_samples_K[:, None])

        self.data["cmb"] = np.zeros((self.instrument.dets.n, self.plan.n_time))

        for band in self.instrument.bands:
            band_mask = self.instrument.dets.band_name == band.name

            band_cmb_power_samples_W = np.trapz(
                y=cmb_brightness * band.passband(1e-9 * test_nu), x=test_nu
            )

            self.data["cmb"][band_mask] = np.interp(
                T_CMB + cmb_temperatures[band_mask],
                cmb_temperature_samples_K,
                band_cmb_power_samples_W,
            )
