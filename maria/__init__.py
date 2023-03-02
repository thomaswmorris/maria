
# -- General packages --
import numpy as np
import scipy as sp
import healpy as hp

from datetime import datetime
from matplotlib import pyplot as plt
from astropy.io import fits

import weathergen
sites = weathergen.sites

# -- Specific packages --
from . import utils
from .config_ScanPattern import *
from .config_Telescope import *

from .object_PLAN import *
from .object_ARRAY import *
from .object_LAM import *

# -- Don't know what to do with this --
def is_isoformat(x):
    try: datetime.fromisoformat(x); return True
    except: return False

# -- The "Call this class to get your mock-ob" part --
class Weobserve():
    def __init__(self, project, skymodel, verbose=True, **kwargs):

        self.verbose = verbose
        self.file_name = skymodel
        self.file_save = project

        # get proper telescope
        self._get_configs(**kwargs)

        #get the atmosphere --> Should do something with the pwv
        self._run_atmos()

        #get the CMB?


        #Get the astronomical signal
        self._get_skyconfig(**kwargs)
        self._get_sky()
        self._savesky()
        
        # save relevant files
        #   Noise map
        #   Model map without noise
        #   Model map with Noise
        #   observation details?

    def _get_configs(self, **kwargs):

        self.meta_data = {   
            'Observatory':     kwargs.get('Observatory', 'AtLAST'),
            'Scanning_patern': kwargs.get('Scanning_patern', 'daisy')
        }

        #get your defaults
        self.ARRAY_CONFIG = OBSERVATORIES[self.meta_data['Observatory']]
        self.PLAN_CONFIG  = SCANNINGPATTERNS[self.meta_data['Scanning_patern']]
        
        #additional telescope request like:
        for k in self.ARRAY_CONFIG.keys():
            if k in kwargs: self.ARRAY_CONFIG[k] = kwargs.get(k)
        
        #additional observational request like:
        for k in self.PLAN_CONFIG.keys():
            if k in kwargs: self.PLAN_CONFIG[k] = kwargs.get(k)
            
        #   integration time --> integration
        #   pwv --> 0.5 mm
        #   direction --> indirection

    def _run_atmos(self):
            self.lam = LAM(Array(self.ARRAY_CONFIG), 
                            Plan(self.PLAN_CONFIG), 
                            verbose=self.verbose) 

            self.lam.simulate_atmosphere()

    def _get_skyconfig(self, **kwargs):
        hudl = fits.open(self.file_name)
        self.im = hudl[0].data
        self.he = hudl[0].header

        self.sky_data = {
            'inbright':     kwargs.get('inbright', None), #assuming written in Jy/pixel
            'incell':       kwargs.get('incell',   self.he['CDELT1']), #assuming written in degree
            'inwidth':      kwargs.get('inwidth',  None), #assuming written in Hz --> for the spectograph...
        }

        if self.sky_data['inbright'] != None: self.im = self.im/np.nanmax(self.im) * self.sky_data['inbright']

    #need to rewrite this
    def _get_sky(self,):

        map_res = np.radians(self.sky_data['incell']) #changed to the hyper parameter
        map_nx, map_ny = self.im.shape

        map_x = map_res * map_nx * np.linspace(-0.5, 0.5, map_nx)
        map_y = map_res * map_ny * np.linspace(-0.5, 0.5, map_ny)

        map_X, map_Y = np.meshgrid(map_x, map_y, indexing='ij')
        map_azim, map_elev = utils.from_xy(map_X, map_Y, self.lam.azim.mean(), self.lam.elev.mean()) #performed the lam.l to lam. elev and lam.b to lam.azim change here

        map_im = np.zeros(hp.pixelfunc.nside2npix(len(self.im))) #### hard coded imsize --> changed to len(self.im)
        map_im[hp.pixelfunc.ang2pix(2048, np.pi/2 - map_elev, map_azim)] = self.im

        lam_x, lam_y = utils.to_xy(self.lam.elev, self.lam.azim, self.lam.elev.mean(), self.lam.azim.mean())
        map_data = sp.interpolate.RegularGridInterpolator((map_x, map_y), self.im, bounds_error=False, fill_value=0)((lam_x, lam_y))
        map_data = hp.get_interp_val(map_im, np.pi/2 - self.lam.elev, self.lam.azim)

        #what is this?
        x_bins = np.arange(map_X.min(), map_X.max(), 8 * map_res)
        y_bins = np.arange(map_Y.min(), map_Y.max(), 8 * map_res)
        
        true_map = sp.stats.binned_statistic_2d(map_X.ravel(), 
                                map_Y.ravel(),
                                self.im.ravel(),
                                statistic='mean',
                                bins=(x_bins, y_bins))[0]

        filtered_map = sp.stats.binned_statistic_2d(lam_x.ravel(), 
                                lam_y.ravel(),
                                map_data.ravel(),
                                statistic='mean',
                                bins=(x_bins, y_bins))[0]
        
        total_map = sp.stats.binned_statistic_2d(lam_x.ravel(), 
                          lam_y.ravel(),
                          (map_data + self.lam.atm_power).ravel(), #should add CMB here, changed atm_power_data to chagned atm_power
                          statistic='mean',
                          bins=(x_bins, y_bins))[0]    

        noise_map = sp.stats.binned_statistic_2d(lam_x.ravel(), 
                          lam_y.ravel(),
                          self.lam.atm_power.ravel(), #should add CMB here, changed atm_power_data to chagned atm_power
                          statistic='mean',
                          bins=(x_bins, y_bins))[0]     
        
        self.noisemap     = noise_map
        self.filteredmap  = filtered_map
        self.mockobs      = total_map

        plt.figure()
        plt.imshow(true_map)
        plt.colorbar(location='bottom', shrink=0.8)
        plt.show()

        plt.figure()
        plt.imshow(self.noisemap)
        plt.colorbar(location='bottom', shrink=0.8)
        plt.show()

        plt.figure()
        plt.imshow(self.filteredmap)
        plt.colorbar(location='bottom', shrink=0.8)
        plt.show()

        plt.figure()
        plt.imshow(self.mockobs)
        plt.colorbar(location='bottom', shrink=0.8)
        plt.show()

        
    def _savesky(self,):
        ...

