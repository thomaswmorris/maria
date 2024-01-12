import os

import h5py

from ... import utils
from ...weather import InvalidRegionError, all_regions

here, this_filename = os.path.split(__file__)

SPECTRUM_DATA_URL_BASE = (
    'https://github.com/thomaswmorris/maria/raw/master/maria/atmosphere/spectra/data'
)

SPECTRUM_DATA_CACHE_DIR = '/tmp/maria_cache/spectra'
MAX_CACHE_AGE_SECONDS = 86400


class AtmosphericSpectrum:
    def __init__(self, region):
        """
        A dataclass to hold spectra as attributes.
        """

        if region not in all_regions:
            raise InvalidRegionError(region)

        self.source_path = f'{here}/data/{region}.h5'

        # if the data isn't in the module, use the cache
        if os.path.exists(self.source_path):
            self.from_cache = False
        else:
            self.source_path = f'{SPECTRUM_DATA_CACHE_DIR}/{region}.h5'
            utils.io.fetch_cache(
                source_url=f'{SPECTRUM_DATA_URL_BASE}/{region}.h5',
                cache_path=self.source_path,
            )
            self.from_cache = True

        with h5py.File(self.source_path, 'r') as f:
            self.side_nu_GHz = f['side_nu_GHz'][:].astype(float)
            self.side_elevation_deg = f['side_elevation_deg'][:].astype(float)
            self.side_line_of_sight_pwv_mm = f['side_line_of_sight_pwv_mm'][:].astype(
                float
            )
            self.temperature_rayleigh_jeans_K = f['temperature_rayleigh_jeans_K'][
                :
            ].astype(float)
            self.phase_delay_um = f['phase_delay_um'][:].astype(float)
