import glob
import re
import os
import weathergen
import h5py
import typing
from . import utils
from dataclasses import dataclass

here, this_filename = os.path.split(__file__)

REGIONS_WITH_SPECTRA = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
REGIONS_WITH_WEATHER = list(weathergen.regions.index)
SUPPORTED_REGIONS = list(set(REGIONS_WITH_SPECTRA) & set(REGIONS_WITH_WEATHER))

regions = weathergen.regions.loc[SUPPORTED_REGIONS].sort_index()

SITE_CONFIGS = utils.io.read_yaml(f"{here}/configs/sites.yml")
SITE_PARAMS = set()
for key, config in SITE_CONFIGS.items():
    SITE_PARAMS |= set(config.keys())

class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        super().__init__(f"The site \'{invalid_site}\' is not in the database of default sites. "
        f"Default sites are:\n\n{sorted(list(SITE_CONFIGS.keys()))}")

class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        region_string = regions.to_string(columns=['location', 'country', 'latitude', 'longitude', 'altitude'])
        super().__init__(f"The region \'{invalid_region}\' is not supported. Supported regions are:\n\n{region_string}")

def get_site_config(site_name, **kwargs):
    if not site_name in SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    SITE_CONFIG = SITE_CONFIGS[site_name].copy()
    for k, v in kwargs.items():
        SITE_CONFIG[k] = v
    return SITE_CONFIG

def get_site(site_name, **kwargs):
    return Site(**get_site_config(site_name, **kwargs))


class AtmosphericSpectrum:
    def __init__(self, filepath):
        """
        A dataclass to hold spectra as attributes
        """
        with h5py.File(filepath, "r") as f:

            self.nu             = f["nu_Hz"][:]
            self.side_elevation = f["side_elevation_deg"][:]
            self.side_pwv       = f["side_zenith_pwv_mm"][:]
            self.trj            = f["temperature_rayleigh_jeans_K"][:]
            self.phase_delay    = f["phase_delay_um"][:]

@dataclass
class Site:

    description: str = "",
    region: str = "princeton",
    altitude: float = 62, # in meters
    seasonal: bool = True,
    diurnal: bool = True,
    latitude: float = 40.3522, # in degrees
    longitude: float = -74.6519, # in degrees
    weather_quantiles: dict = {},
    pwv_rms: float = 100, # in microns

    """
    A class containing time-ordered pointing data. Pass a supported site (found at weathergen.sites),
    and a height correction if needed.
    """

    def __post_init__(self):

        if not self.region in regions.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is None:
            self.longitude = regions.loc[self.region].longitude

        if self.latitude is None:
            self.latitude = regions.loc[self.region].latitude

        if self.altitude is None:
            self.altitude = regions.loc[self.region].altitude

        spectrum_filepath = f"{here}/spectra/{self.region}.h5"
        self.spectrum = AtmosphericSpectrum(filepath=spectrum_filepath) if os.path.exists(spectrum_filepath) else None