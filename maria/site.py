import os
import pandas as pd
from . import utils
from .weather import regions
from dataclasses import dataclass, field

here, this_filename = os.path.split(__file__)

# REGIONS_WITH_SPECTRA = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
# REGIONS_WITH_WEATHER = list(weathergen.regions.index)
# SUPPORTED_REGIONS = list(set(REGIONS_WITH_SPECTRA) & set(REGIONS_WITH_WEATHER))

SITE_CONFIGS = utils.io.read_yaml(f"{here}/configs/sites.yml")
SITE_PARAMS = set()
for key, config in SITE_CONFIGS.items():
    SITE_PARAMS |= set(config.keys())

sites = pd.DataFrame(SITE_CONFIGS).T

class InvalidSiteError(Exception):
    def __init__(self, invalid_site):
        sites_string = sites.to_string(columns=['description', 'region', 'latitude', 'longitude', 'altitude', 'documentation'])
        super().__init__(f"The site \'{invalid_site}\' is not supported. Supported sites are:\n\n{sites_string}")

class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        regions_string = regions.to_string(columns=['location', 'country', 'latitude', 'longitude'])
        super().__init__(f"The region \'{invalid_region}\' is not supported. Supported regions are:\n\n{regions_string}")

def get_site_config(site_name="APEX", **kwargs):
    if not site_name in SITE_CONFIGS.keys():
        raise InvalidSiteError(site_name)
    SITE_CONFIG = SITE_CONFIGS[site_name].copy()
    for k, v in kwargs.items():
        SITE_CONFIG[k] = v
    return SITE_CONFIG

def get_site(site_name="APEX", **kwargs):
    return Site(**get_site_config(site_name, **kwargs))


@dataclass
class Site:

    description: str = ""
    region: str = "princeton"
    altitude: float = None # in meters
    seasonal: bool = True
    diurnal: bool = True
    latitude: float = None # in degrees
    longitude: float = None # in degrees
    weather_quantiles: dict = field(default_factory=dict)
    pwv_rms_frac: float = 0.03 # as a fraction of the total
    documentation: str = ''

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