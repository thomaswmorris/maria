import numpy as np

import os
import glob

from datetime import datetime, timedelta

from . import utils

here, this_filename = os.path.split(__file__)

POINTING_CONFIGS = utils.io.read_yaml(f"{here}/configs/pointings.yml")
POINTING_PARAMS = set()
for key, config in POINTING_CONFIGS.items():
    POINTING_PARAMS |= set(config.keys())

class UnsupportedPointingError(Exception):
    def __init__(self, invalid_pointing):
        super().__init__(f"The site \'{invalid_pointing}\' is not in the database of default pointings. "
        f"Default pointings are:\n\n{sorted(list(POINTING_CONFIGS.keys()))}")

def get_pointing_config(pointing_name, **kwargs):
    if pointing_name not in POINTING_CONFIGS.keys():
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
        for key, default_value in POINTING_CONFIGS["default"].items():
            setattr(self, key, kwargs.get(key, default_value))

        # make sure that self.start_datetime exists, and that it's a datetime.datetime object
        if not hasattr(self, 'start_time'):
            self.start_time = datetime.now().timestamp()
        self.start_datetime = utils.datetime_handler(self.start_time)

        # make self.end_datetime
        if hasattr(self, 'end_time'): 
            self.end_datetime = utils.datetime_handler(self.end_time)
        else:
            self.end_datetime = self.start_datetime + timedelta(seconds=self.integration_time)
        
        self.time_min = self.start_datetime.timestamp()
        self.time_max = self.end_datetime.timestamp()
        self.dt = 1 / self.sample_rate

        if self.pointing_units == "degrees":
            self.scan_center = np.radians(self.scan_center)
            self.scan_radius = np.radians(self.scan_radius)

        self.time = np.arange(self.time_min, self.time_max, self.dt)
        self.n_time = len(self.time)

        time_ordered_pointing = utils.get_pointing(self.time,
                                                    scan_center = self.scan_center, # a lon/lat in some frame
                                                    pointing_frame = self.pointing_frame, # the frame, one of "az_el", "ra_dec", "galactic"
                                                    scan_radius = self.scan_radius,
                                                    scan_period = self.scan_period,
                                                    scan_pattern="daisy",
                                                    )

        if self.pointing_frame == "ra_dec":
            self.ra, self.dec = time_ordered_pointing
        elif self.pointing_frame == "az_el":
            self.az, self.el = time_ordered_pointing
        elif self.pointing_frame == "dx_dy":
            self.dx, self.dy = time_ordered_pointing

