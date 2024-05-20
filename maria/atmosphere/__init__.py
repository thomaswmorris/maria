import os
from datetime import datetime

from .spectrum import Spectrum
from .weather import Weather  # noqa F401

here, this_filename = os.path.split(__file__)

SPECTRA_DATA_DIRECTORY = f"{here}/data"
SPECTRA_DATA_CACHE_DIRECTORY = "/tmp/maria-data/atmosphere/spectra"
SPECTRA_DATA_URL_BASE = "https://github.com/thomaswmorris/maria-data/raw/master/atmosphere/spectra"  # noqa F401
CACHE_MAX_AGE_SECONDS = 30 * 86400


class Atmosphere:
    def __init__(
        self,
        t: float = None,
        region: str = "princeton",
        altitude: float = None,
        weather_quantiles: dict = {},
        weather_kwargs: dict = {},
        weather_source: str = "era5",
        spectrum_source: str = "am",
        pwv_rms_frac: float = 0.03,
    ):
        t = t or datetime.utcnow().timestamp()

        self.spectrum = Spectrum(
            region=region,
            source=spectrum_source,
        )

        self.weather = Weather(
            t=t,
            region=region,
            altitude=altitude,
            quantiles=weather_quantiles,
            override=weather_kwargs,
            source=weather_source,
        )

        self.pwv_rms_frac = pwv_rms_frac
