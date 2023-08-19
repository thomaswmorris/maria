import numpy as np

import os

from datetime import datetime, timedelta

from . import utils

here, this_filename = os.path.split(__file__)

POINTING_CONFIGS = utils.read_yaml(f'{here}/configs/pointings.yml')
POINTINGS = list((POINTING_CONFIGS.keys()))

class UnsupportedPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(f"The site \'{invalid_pointing}\' is not in the database of default pointings. "
        f"Default pointings are:\n\n{sorted(list(POINTING_CONFIGS.keys()))}")

def get_pointing_config(pointing_name, **kwargs):
    if not pointing_name in POINTING_CONFIGS.keys():
        raise UnsupportedPointingError(pointing_name)
    POINTING_CONFIG = POINTING_CONFIGS[pointing_name].copy()
    for k, v in kwargs.items():
        POINTING_CONFIG[k] = v
    return POINTING_CONFIG


def get_pointing(pointing_name, **kwargs):
    return Pointing(**get_pointing_config(pointing_name, **kwargs))


class Pointing:

    """
    A class containing time-ordered pointing data.
    """

    @staticmethod
    def validate_pointing_kwargs(kwargs):
        """
        Make sure that we have all the ingredients to produce the pointing data.
        """
        if ('end_time' not in kwargs.keys()) and ('integration_time' not in kwargs.keys()):
            raise ValueError('One of "end_time" or "integration_time" must be in the pointing kwargs.')

    def __init__(self, **kwargs):

        # these are all required kwargs. if they aren't in the passed kwargs, get them from here.
        DEFAULT_POINTING_CONFIG = {
        "integration_time": 600,
        "pointing_center": [0, 90],
        "pointing_frame": "az_el",
        "pointing_units": "degrees",
        "scan_pattern": "back-and-forth",  
        "scan_period": 60,  
        "sample_rate": 20, 
        }

        for key, val in kwargs.items():
            setattr(self, key, val)

        for key, val in DEFAULT_POINTING_CONFIG.items():
            if not key in kwargs.keys():
                setattr(self, key, val)

        #self.validate_pointing_kwargs(kwargs)

        # make sure that self.start_datetime exists, and that it's a datetime.datetime object
        if not hasattr(self, 'start_time'):
            self.start_time = datetime.now().timestamp()
        self.start_datetime = utils.datetime_handler(self.start_time)

        # make self.end_datetime
        if hasattr(self, 'end_time'): 
            self.end_datetime = utils.datetime_handler(self.end_time)
        else:
            self.end_datetime = self.start_datetime + timedelta(seconds=self.integration_time)
        
        self.unix_min = self.start_datetime.timestamp()
        self.unix_max = self.end_datetime.timestamp()
        self.dt = 1 / self.sample_rate

        if self.pointing_units == "degrees":
            self.pointing_center = np.radians(self.pointing_center)
            self.pointing_throws = np.radians(self.pointing_throws)

        self.unix = np.arange(self.unix_min, self.unix_max, self.dt)
        self.n_time = len(self.unix)

        time_ordered_pointing = utils.get_pointing(
            self.unix,
            self.scan_period,
            self.pointing_center,
            self.pointing_throws,
            self.scan_pattern,
        )

        if self.pointing_frame == "ra_dec":
            self.ra, self.dec = time_ordered_pointing
        elif self.pointing_frame == "az_el":
            self.az, self.el = time_ordered_pointing
        elif self.pointing_frame == "dx_dy":
            self.dx, self.dy = time_ordered_pointing

