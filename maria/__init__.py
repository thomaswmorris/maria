

import numpy as np
import scipy as sp

from tqdm import tqdm

import warnings

from scipy import signal, spatial
from numpy import linalg as la

from importlib import resources

import time as ttime

from . import tools

import weathergen



DEFAULT_ARRAY_CONFIG = {'shape' : 'hex',      # shape of detector arrangement
                        'n_det' : 64,         # number of detectors
                          'fov' : 2,          # maximum detector separation [degrees]
                 'primary_size' : 5,          # size of primary mirror [meters]
                  'white_noise' : 0,          # maximum span of array
                'optical_model' : 'diff_lim', # 
                        'bands' : 150e9,
                   'bandwidths' : 10e9,
                
                       }

class array():

    def __init__(self, config={}):

        self.config = {}
        self.put(DEFAULT_ARRAY_CONFIG)
        self.put(config)

    def put(self, config, verbose=False):

        # select an offset mode
        self.offset_mode = 'auto'
        if np.isin(['offsets'], list(config.keys())).all(): self.offset_mode = 'manual'
        else: pass # raise error here

        if verbose: print(self.offset_mode)
        
        # select a passband mode
        self.passband_mode = 'auto'
        if np.isin(['passbands'], list(config.keys())).all(): self.passband_mode = 'manual'
        else: pass # raise error here

        for key, val in config.items(): 
            self.config[key] = val
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

        self.compute()

    def compute(self):

        # compute detector offsets

        if self.offset_mode == 'auto':
            self.offsets = tools.make_array(self.shape, self.fov, self.n_det)


        else:
            self.array_shape = 'custom'
            self.n_det = len(self.offsets)
            self.fov = np.sqrt(np.sum([np.square(np.subtract.outer(e, e)) for e in self.offsets.T])).max()

        self.offsets *= np.pi / 180
        # compute passbands

        self.hull = sp.spatial.qhull.ConvexHull(self.offsets)

        self.offset_x, self.offset_y = self.offsets.T
        self.offset_z = self.offset_x + 1j * self.offset_y
        self.offset_r, self.offset_p = np.abs(self.offset_z), np.angle(self.offset_z)

        if self.passband_mode == 'auto':

            self.bands      = np.array([self.bands]).ravel()
            self.bandwidths = np.array([self.bandwidths]).ravel()

            if len(self.bands) == 1:      self.bands = np.repeat(self.bands, self.n_det)
            if len(self.bandwidths) == 1: self.bandwidths = np.repeat(self.bandwidths, self.n_det)

            self.nu = np.arange((self.bands - 0.75 * self.bandwidths).min(), (self.bands + 0.75 * self.bandwidths).max(), 2e9)

            self.passbands  = np.c_[[tools.get_passband(self.nu, nu_0, nu_w, order=8) for nu_0, nu_w in zip(self.bands, self.bandwidths)]]
            self.passbands /= self.passbands.sum(axis=1)[:,None]
        

        # compute beams

        if self.optical_model == 'diff_lim':

            self.get_beam_waist = lambda z, w_0, f : w_0 * np.sqrt(1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0)))

            gauss_8 = lambda r, r_fwhm : np.exp(np.log(0.5)*np.abs(r/r_fwhm)**8)

            self.beam_func = gauss_8
    

DEFAULT_PLAN_CONFIG = {  'duration' : 120,    # shape of detector arrangement
                      'sample_rate' : 20,     # number of detectors
                      'scan_period' : 30,     # maximum detector separation [degrees]
                        'scan_type' : 'baf', 
                               'az' : 0,
                               'el' : 45,
                         'az_throw' : 10,
                         'el_throw' : 0,
                      }


class plan():

    '''
    'baf' : back-and-forth 
    'box' : box scan           
    'lis' : lissajous      
    '''

    def __init__(self, config={}):

        self.config = {}
        self.put(DEFAULT_PLAN_CONFIG)
        self.put(config)

    def put(self, config, verbose=False):

        self.pointing_mode = 'auto'

        if np.isin(['time', 'azim', 'elev'], list(config.keys())).all(): self.offset_mode = 'manual'
        else: pass # raise error here

        if verbose: print(self.pointing_mode)

        # Overwrite new values

        for key, val in config.items(): 
            self.config[key] = val
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

        self.compute()

    def compute(self):

        self.dt = 1 / self.sample_rate
        
        if self.pointing_mode == 'auto':

            self.time  = np.arange(0, self.duration, self.dt)
            self.phase = 2 * np.pi * (self.time / self.scan_period % 1)

            if self.scan_type == 'baf': 

                self.deg_azim, self.deg_elev = tools.baf_pointing(self.phase, self.az, self.el, self.az_throw, self.el_throw)

            if self.scan_type == 'box': 

                self.deg_azim, self.deg_elev = tools.box_pointing(self.phase, self.az, self.el, self.az_throw, self.el_throw)

            self.c_azim, self.c_elev = np.radians(self.deg_azim), np.radians(self.deg_elev)

        
        self.c_x, self.c_y = tools.to_xy(self.c_azim, self.c_elev, self.c_azim.mean(), self.c_elev.mean())

        self.c_x_v = np.gradient(self.c_x) / self.dt
        self.c_y_v = np.gradient(self.c_y) / self.dt


DEFAULT_SITE_CONFIG = {'time_UTC' : 0,
                       'latitude' : -23.5,
                      'longitude' : -67.5,
                       'altitude' : 5e3,
                         'region' : 'chajnantor',
                        }

class site():

    def __init__(self, config={}):

        self.config = {}
        self.put(DEFAULT_SITE_CONFIG)
        self.put(config)

    def put(self, config, verbose=False):

        self.site_mode = 'auto'

        if np.isin(['latitude', 'longitude', 'altitude'], list(config.keys())).all(): self.site_mode = 'manual'
        else: pass # raise error here

        if verbose: print(self.pointing_mode)
        # Overwrite new values

        for key, val in config.items(): 
            self.config[key] = val
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

        self.compute()

    def compute(self):

        self.weather = weathergen.generate(region=self.region, time=self.time_UTC, method='random')



DEFAULT_LAM_CONFIG = {'min_depth' : 1000,
                      'max_depth' : 2000,
                       'n_layers' : 2,
                       'min_beam_res' : 4,
                       
                       }

class lam():
    
    def __init__(self, array, plan, site, config={}, verbose=False):

        self.array, self.plan, self.site = array, plan, site

        self.config = {}
        self.put(DEFAULT_LAM_CONFIG)
        self.put(config)
        
        for key, val in config.items(): 
            self.config[key] = val
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

    def put(self, config, verbose=False):

        self.site_mode = 'auto'

        if np.isin(['latitude', 'longitude', 'altitude'], list(config.keys())).all(): self.site_mode = 'manual'
        else: pass # raise error here

        if verbose: print(self.pointing_mode)
        # Overwrite new values

        for key, val in config.items(): 
            self.config[key] = val
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

        #self.initialize()

    def initialize(self, verbose=False):

        self.depths = np.linspace(self.min_depth, self.max_depth, self.n_layers)
        self.thicks = np.gradient(self.depths)

        self.waists = self.array.get_beam_waist(self.depths[:,None], self.array.primary_size, self.array.nu[None,:])

        self.angular_waists = self.waists / self.depths[:,None]

        self.min_ang_res = self.angular_waists / self.min_beam_res

        self.heights = self.depths[:,None] * np.sin(self.plan.c_elev[None,:])
        self.wvmd = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['water_density'])
        self.temp = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['temperature'])
        self.var_scaling = np.square(self.wvmd * self.temp)
        self.rel_scaling = np.sqrt(self.var_scaling / self.var_scaling.sum(axis=0)[None,:])
        self.lay_scaling = 1e-2 * self.site.weather['pwv'] * self.rel_scaling * self.thicks[:, None] / self.thicks.sum()


        self.theta_x = self.array.offset_x[:, None] + self.plan.c_x[None, :] 
        self.theta_y = self.array.offset_y[:, None] + self.plan.c_y[None, :]

        self.theta_z = self.theta_x + 1j * self.theta_y

        self.w_e = np.interp(self.depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_east'])
        self.w_n = np.interp(self.depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_north'])

        self.w_b = np.arctan2(self.w_e, self.w_n)
        self.w_s = np.sqrt(np.square(self.w_e) + np.square(self.w_n))

        self.w_v_x = (+ self.w_e * np.cos(self.plan.c_azim[None,:]) - self.w_n * np.sin(self.plan.c_azim[None,:])) / self.depths[:,None]
        self.w_v_y = (- self.w_e * np.sin(self.plan.c_azim[None,:]) + self.w_n * np.cos(self.plan.c_azim[None,:])) / self.depths[:,None] * np.sin(self.plan.c_elev[None,:])
        
        ### These are empty lists we need to fill with chunky parameters (they won't fit together!) for each layer. 
        self.para, self.orth, self.X, self.Y, self.P, self.O = [], [], [], [], [], []
        self.n_para, self.n_orth, self.lay_ang_res, self.genz, self.AR_samples = [], [], [], [], []

        #self.rel_theta_z = self.theta_x + np.cumsum(self.w_v_x * self.plan.dt) + 1j# * (self.theta_y + np.cumsum(self.w_v_y * self.plan.dt))

        self.rel_c_x = self.plan.c_x[None,:] + np.cumsum(self.w_v_x * self.plan.dt, axis=1) 
        self.rel_c_y = self.plan.c_y[None,:] + np.cumsum(self.w_v_y * self.plan.dt, axis=1) 
        
        self.zop = np.zeros((self.theta_x.shape),dtype=complex)
        self.p   = np.zeros((self.theta_x.shape))
        self.o   = np.zeros((self.theta_x.shape))

        self.MARA = []
        self.outer_scale = 1e2
        self.ang_outer_scale = self.outer_scale / self.depths
        
        self.theta_edge_z = []

        radius_sample_prop = 1.5
        beam_tol = 1e-1

        max_layer_beam_radii = 0.5 * self.angular_waists.max(axis=1)

        self.padded_radius = (radius_sample_prop + beam_tol) * max_layer_beam_radii + self.array.offset_r.max()
        
        for i_l, depth in enumerate(self.depths):

            hull = sp.spatial.qhull.ConvexHull(np.c_[self.rel_c_x[i_l], self.rel_c_y[i_l]])
            h_x, h_y = hull.points[hull.vertices].T; h_z = h_x + 1j * h_y
            layer_hull_theta_z = h_z * (np.abs(h_z) + self.padded_radius[i_l]) / np.abs(h_z)

            self.MARA.append(tools.get_MARA(layer_hull_theta_z.ravel()))
                        
            # an efficient way to compute the minimal observing area that we need to generate
            self.theta_edge_z.append(layer_hull_theta_z)

            RZ = layer_hull_theta_z * np.exp(1j*self.MARA[-1])
            
            para_min, para_max = np.real(RZ).min(), np.real(RZ).max()
            orth_min, orth_max = np.imag(RZ).min(), np.imag(RZ).max()
            
            para_center, orth_center = (para_min + para_max)/2, (orth_min + orth_max)/2
            para_radius, orth_radius = (para_max - para_min)/2, (orth_max - orth_min)/2
    
            n_orth_min = 64
            n_orth_max = 1024

            lay_ang_res = np.minimum(self.min_ang_res[i_l].min(), 2 * orth_radius / (n_orth_min - 1))
            lay_ang_res = np.maximum(lay_ang_res, 2 * orth_radius / (n_orth_max - 1))

            
            self.lay_ang_res.append(lay_ang_res)
            
            para_ = para_center + np.arange(-para_radius,para_radius+.5*lay_ang_res,lay_ang_res)
            orth_ = orth_center + np.arange(-orth_radius,orth_radius+.5*lay_ang_res,lay_ang_res)
            
            self.PARA_SPACING = np.gradient(para_).mean()
            self.para.append(para_), self.orth.append(orth_)
            self.n_para.append(len(para_)), self.n_orth.append(len(orth_))
        
            ORTH_,PARA_ = np.meshgrid(orth_,para_)
            
            self.genz.append(np.exp(-1j*self.MARA[-1]) * (PARA_[0] + 1j*ORTH_[0] - self.PARA_SPACING) )
            layer_ZOP = np.exp(-1j*self.MARA[-1]) * (PARA_ + 1j*ORTH_) 
            
            self.X.append(np.real(layer_ZOP)), self.Y.append(np.imag(layer_ZOP))
            self.O.append(ORTH_), self.P.append(PARA_)
            
            
            
            self.zop[i_l] = self.theta_z[i_l] * np.exp(1j*self.MARA[-1]) 
            self.p[i_l], self.o[i_l] = np.real(self.zop[i_l]), np.imag(self.zop[i_l])

            cov_args = (1,1)
            
            para_i, orth_i = [],[]
            for ii,i in enumerate(np.r_[0,2**np.arange(np.ceil(np.log(self.n_para[-1])/np.log(2))),self.n_para[-1]-1]):
                
                #if i * self.ang_res[i_l] > 2 * self.ang_outer_scale[i_l]:
                #    continue
                
                #orth_i.append(np.unique(np.linspace(0,self.n_orth[-1]-1,int(np.maximum(self.n_orth[-1]/(i+1),16))).astype(int)))
                orth_i.append(np.unique(np.linspace(0,self.n_orth[-1]-1,int(np.maximum(self.n_orth[-1]/(4**ii),4))).astype(int)))
                para_i.append(np.repeat(i,len(orth_i[-1])).astype(int))
                
            self.AR_samples.append((np.concatenate(para_i),np.concatenate(orth_i)))
            
            n_cm = len(self.AR_samples[-1][0])
            
            if n_cm > 5000 and show_warnings:
                
                warning_message = f'A very large covariance matrix for layer {i_l+1} (n_side = {n_cm})'
                warnings.warn(warning_message)
            

        self.prec, self.csam, self.cgen, self.A, self.B = [], [], [], [], []
        
        with tqdm(total=len(self.depths),desc='Computing weights') as prog:
            for i_l, (depth, LX, LY, AR, GZ) in enumerate(zip(self.depths,self.X,self.Y,self.AR_samples,self.genz)):
                
                cov_args  = (self.outer_scale / depth, 5/6)
                
                self.prec.append(la.inv(tools.make_2d_covariance_matrix(tools.matern,cov_args,LX[AR],LY[AR])))

                self.cgen.append(tools.make_2d_covariance_matrix(tools.matern,cov_args,np.real(GZ),np.imag(GZ)))
                
                self.csam.append(tools.make_2d_covariance_matrix(tools.matern,cov_args,np.real(GZ),np.imag(GZ),LX[AR],LY[AR],auto=False)) 
                
                self.A.append(np.matmul(self.csam[i_l],self.prec[i_l])) 
                self.B.append(tools.msqrt(self.cgen[i_l]-np.matmul(self.A[i_l],self.csam[i_l].T)))
                
                prog.update(1)

        if verbose:
            print('\n # | depth (m) | beam (m) | beam (\') | sim (m) | sim (\') | rms (mg/m2) | n_cov | orth | para | h2o (mg/m3) | temp (K) | ws (m/s) | wb (deg) |')
            
            for i_l, depth in enumerate(self.depths):
                
                row_string  = f'{i_l+1:2} | {depth:9.01f} | {self.waists[i_l].min():8.02f} | {60*np.degrees(self.angular_waists[i_l].min()):8.02f} | '
                row_string += f'{depth*self.lay_ang_res[i_l]:7.02f} | {60*np.degrees(self.lay_ang_res[i_l]):7.02f} | '
                row_string += f'{1e3*self.lay_scaling[i_l].mean():11.02f} | {len(self.AR_samples[i_l][0]):5} | {self.n_orth[i_l]:4} | '
                row_string += f'{self.n_para[i_l]:4} | {1e3*self.wvmd[i_l].mean():11.02f} | {self.temp[i_l].mean():8.02f} | '
                row_string += f'{self.w_s[i_l].mean():8.02f} | {np.degrees(self.w_b[i_l].mean()+np.pi):8.02f} |'
                print(row_string)

        


    def atmosphere_timestep(self,i): # iterate the i-th layer of atmosphere by one step
        
        self.vals[i] = np.r_[(np.matmul(self.A[i],self.vals[i][self.AR_samples[i]])
                            + np.matmul(self.B[i],np.random.standard_normal(self.B[i].shape[0])))[None,:],self.vals[i][:-1]]

    def generate_atmosphere(self,blurred=False):

        self.vals = [np.zeros(lx.shape, dtype=np.float16) for lx in self.X]
        n_init_   = [n_para for n_para in self.n_para]
        n_ts_     = [n_para for n_para in self.n_para]
        tot_n_init, tot_n_ts = np.sum(n_init_), np.sum(n_ts_)
        #self.gen_data = [np.zeros((n_ts,v.shape[1])) for n_ts,v in zip(n_ts_,self.lay_v_)]

        with tqdm(total=tot_n_init,desc='Generating layers') as prog:
            for i, n_init in enumerate(n_init_):
                for i_init in range(n_init):
                    
                    self.atmosphere_timestep(i)
                    
                    prog.update(1)
                
        
    def sim(self, do_atmosphere=True, 
                  units='mK_CMB', 
                  do_noise=True,
                  split_layers=False,
                  split_bands=False):
        
        self.sim_start = ttime.time()
        self.generate_atmosphere()
        
        #print(self.array.band_weights.shape)
        #temp_data = np.zeros((len(self.atmosphere.depths),self.array.n,self.pointing.nt))
        
        self.epwv = self.site.weather['pwv'] + np.zeros((self.array.n,self.pointing.nt))
        
        self.n_bf, self.beam_filters, self.beam_filter_sides = [], [], []
        
        with tqdm(total=len(self.atmosphere.depths) + len(self.array.nom_band_list),
                  desc='Sampling atmosphere') as prog:
            
            
            for i_l, depth in enumerate(self.atmosphere.depths): 
                
                waist_samples, which_sample = tools.smallest_max_error_sample(self.ang_waists[i_l],max_error=1e-1)
                
                wv_data = np.zeros((self.array.n,self.pointing.nt))
                
                for i_w, w in enumerate(waist_samples):
                #for i_ba, nom_band in enumerate(self.array.nom_band_list):
                    
                    # band-waist mask : which detectors observe bands that are most effectively modeled by this waist?
                    bm = np.isin(self.array.nom_bands,self.array.nom_band_list[which_sample == i_w])
                    
                    # filter the angular atmospheric emission, to account for beams
                    self.n_bf.append(int(np.ceil(.6 * w / self.lay_ang_res[i_l])))
                    self.beam_filter_sides.append(self.lay_ang_res[i_l] * np.arange(-self.n_bf[-1],self.n_bf[-1]+1))
                    self.beam_filters.append(tools.make_beam_filter(self.beam_filter_sides[-1],self.beams.get_window,[w/2]))
                
                    
                    filtered_vals = sp.signal.convolve2d(self.vals[i_l], self.beam_filters[-1], boundary='symm', mode='same')
                    #sigma = .5 * w / self.lay_ang_res[i_l]
                    #print(sigma)
                    #filtered_vals = sp.ndimage.gaussian_filter(self.vals[i_l], sigma=sigma)
                    
                    FRGI = sp.interpolate.RegularGridInterpolator((self.para[i_l],self.orth[i_l]), self.lay_scaling[i_l] * filtered_vals)
                    wv_data[bm] = FRGI((self.pointing.p[i_l][bm],self.pointing.o[i_l][bm]))
                   
                    
                self.epwv += wv_data
                prog.update(1)
