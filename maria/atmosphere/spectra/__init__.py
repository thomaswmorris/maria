import os

import h5py
import requests

from ...weather import InvalidRegionError, all_regions

here, this_filename = os.path.split(__file__)

SPECTRUM_DATA_URL_BASE = (
    "https://github.com/thomaswmorris/maria/raw/master/maria/atmosphere/spectra/data"
)


class AtmosphericSpectrum:
    def __init__(self, region):
        """
        A dataclass to hold spectra as attributes.
        """

        if region not in all_regions:
            raise InvalidRegionError(region)

        self._spectrum_path = f"{here}/data/{region}.h5"

        # download the data as needed
        if not os.path.exists(self._spectrum_path):
            print("getting spectrum data...")
            url = f"{SPECTRUM_DATA_URL_BASE}/{region}.h5"
            r = requests.get(url)
            with open(self._spectrum_path, "wb") as f:
                f.write(r.content)

        with h5py.File(self._spectrum_path, "r") as f:
            self.side_nu_GHz = f["side_nu_GHz"][:].astype(float)
            self.side_elevation_deg = f["side_elevation_deg"][:].astype(float)
            self.side_line_of_sight_pwv_mm = f["side_line_of_sight_pwv_mm"][:].astype(
                float
            )
            self.temperature_rayleigh_jeans_K = f["temperature_rayleigh_jeans_K"][
                :
            ].astype(float)
            self.phase_delay_um = f["phase_delay_um"][:].astype(float)
