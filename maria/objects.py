



from importlib import resources

from . import tools





DEFAULT_CONFIG = {'array_shape' : 'hex',    # shape of detector arrangement
                        'n_det' : 64,       # number of detectors
                          'fov' : 2,       # maximum detector separation [degrees]
                 'primary_size ' : 1.5e11,   # size of primary mirror [meters]
                  'white_noise' : 0,        # maximum span of array
                'optical_model' : '',
                }

class Array():

    def __init__(self, config=DEFAULT_CONFIG):

        for k, v in config.items():

            setattr(self, k, v)

        if not 'offsets' in dir(self):

            self.offsets = tools.make_array(self.array_shape, self.fov, self.n_det)


class Site():

    def __init__(self, config=DEFAULT_CONFIG):

        self.time_UTC  = 0
        self.region    = 'default'
        self.latitude  = 0
        self.longitude = 0
        self.altitude  = 0




class LAM(Array, Site):

    pass


class Model(Array, Site):

    def __init__(self, name=None, config=DEFAULT_CONFIG):

        if not name is None:
            with resources.path("maria", "site_info.csv") as f:
                self.site_df = pd.read_csv(f, index_col=0)

        #Site.__init__()




