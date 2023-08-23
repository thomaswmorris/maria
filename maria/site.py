import glob
import re
import os
import weathergen
from . import utils

here, this_filename = os.path.split(__file__)

REGIONS_WITH_SPECTRA = [re.findall(rf"{here}/spectra/(.+).h5", filepath)[0] for filepath in glob.glob(f"{here}/spectra/*.h5")]
REGIONS_WITH_WEATHER = list(weathergen.regions.index)
SUPPORTED_REGIONS = list(set(REGIONS_WITH_SPECTRA) & set(REGIONS_WITH_WEATHER))

regions = weathergen.regions.loc[SUPPORTED_REGIONS].sort_index()

SITE_CONFIGS = utils.read_yaml(f'{here}/configs/sites.yml')

SITES = list((SITE_CONFIGS.keys()))

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


class Site:

    """
    A class containing time-ordered pointing data. Pass a supported site (found at weathergen.sites),
    and a height correction if needed.
    """

    def __init__(self, **kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)

        if not self.region in regions.index.values:
            raise InvalidRegionError(self.region)

        if self.longitude is None:
            self.longitude = regions.loc[self.region].longitude

        if self.latitude is None:
            self.latitude = regions.loc[self.region].latitude

        if self.altitude is None:
            self.altitude = regions.loc[self.region].altitude

        
    def __repr__(self):
        
        return f"{self.region}"