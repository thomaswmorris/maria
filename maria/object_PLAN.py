from . import utils

import numpy as np

class Plan():

    '''
    'back_and_forth' : back-and-forth        
    'daisy'          : lissajous daisy
    '''

    def __init__(self, config):

        self.config = config
        self.put(config)

    def put(self, config, verbose=False):

        for key, val in config.items(): 
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

        self.start_time = utils.datetime_handler(self.start_time)
        self.end_time   = utils.datetime_handler(self.end_time)

        self.compute()

    def compute(self):

        self.dt = 1 / self.sample_rate

        self.t_min = self.start_time.timestamp()
        self.t_max = self.end_time.timestamp()

        self.unix   = np.arange(self.t_min, self.t_max, self.dt)  
        self.coords = utils.get_pointing(self.unix, self.scan_period, self.coord_center, self.coord_throw, self.scan_pattern, self.scan_options)