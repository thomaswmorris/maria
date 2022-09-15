

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


from os import path






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

        self.hull = sp.spatial.qhull.ConvexHull(self.offsets)

        self.x, self.y = self.offsets.T
        self.r, self.p = np.abs(self.x + 1j * self.y), np.angle(self.x + 1j * self.y)

        if self.passband_mode == 'auto':

            self.bands      = np.array([self.bands]).ravel()
            self.bandwidths = np.array([self.bandwidths]).ravel()
            self.ubands     = np.unique(self.bands)

            if len(self.bands) == 1:      self.bands = np.repeat(self.bands, self.n_det)
            if len(self.bandwidths) == 1: self.bandwidths = np.repeat(self.bandwidths, self.n_det)

            self.nu = np.arange((self.bands - 0.75 * self.bandwidths).min(), (self.bands + 0.75 * self.bandwidths).max(), 1e9)

            self.passbands  = np.c_[[tools.get_passband(self.nu, nu_0, nu_w, order=8) for nu_0, nu_w in zip(self.bands, self.bandwidths)]]
            
            good_nu = self.passbands.max(axis=0) > 1e-4
            self.nu = self.nu[good_nu]

            self.passbands  = self.passbands[:,good_nu]
            self.passbands /= self.passbands.sum(axis=1)[:,None]

        # compute beams

        if self.optical_model == 'diff_lim':

            self.get_beam_waist = lambda z, w_0, f : w_0 * np.sqrt(1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0)))

            gauss_8 = lambda r, r_fwhm : np.exp(np.log(0.5)*np.abs(r/r_fwhm)**8)

            self.beam_func = gauss_8

    def make_filter(self, waist, res, func, width_per_waist=1.2):
    
        filter_width = width_per_waist * waist
        n_filter = 2 * int(np.ceil(0.5 * filter_width / res)) + 1

        filter_side = 0.5 * np.linspace(-filter_width, filter_width, n_filter)

        FILTER_X, FILTER_Y = np.meshgrid(filter_side, filter_side, indexing='ij')
        FILTER_R = np.sqrt(np.square(FILTER_X) + np.square(FILTER_Y))

        FILTER  = func(FILTER_R, 0.5 * waist)
        FILTER /= FILTER.sum()
        
        return FILTER
    
    def separate_filter(self, F, tol=1e-2):
        
        u, s, v = la.svd(F); eff_filter = 0
        for m, (_u, _s, _v) in enumerate(zip(u.T, s, v)):

            eff_filter += _s * np.outer(_u, _v)
            if np.abs(F - eff_filter).sum() < tol: break

        return u.T[:m+1], s[:m+1], v[:m+1]

    def separably_filter(self, M, F, tol=1e-2):

        u, s, v = self.separate_filter(F, tol=tol)

        filt_M = 0
        for _u, _s, _v in zip(u, s, v):

            filt_M += _s * sp.ndimage.filters.convolve1d(sp.ndimage.filters.convolve1d(M.astype(float), _u, axis=0), _v, axis=1)

        return filt_M

    

DEFAULT_PLAN_CONFIG = {  'duration' : 120,    # shape of detector arrangement
                      'sample_rate' : 20,     # number of detectors
                      'scan_period' : 30,     # maximum detector separation [degrees]
                        'plan_type' : 'baf', 
                     'plan_options' : {},
                        'az_center' : 0,
                        'el_center' : 45,
                         'az_throw' : 10,
                         'el_throw' : 0,
                      }


class plan():

    '''
    'baf'       : back-and-forth 
    'box'       : box scan           
    'lissajous' : lissajous box
    'rose_3'    : three-petaled rose  
    'rose_12'   : twelve-petaled rose
    '''


    def __init__(self, config={}):

        self.config = {}
        self.put(DEFAULT_PLAN_CONFIG, verbose=False)
        if not config == {}: self.put(config)

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
            
            self.deg_c_azim, self.deg_c_elev = tools.get_pointing(self.time, self.scan_period, self.az_center, self.az_throw, self.el_center, self.el_throw, 
                                                                  self.plan_type, self.plan_options)

            self.c_azim, self.c_elev = np.radians(self.deg_c_azim), np.radians(self.deg_c_elev)
        
        self.c_x, self.c_y = tools.to_xy(self.c_azim, self.c_elev, self.c_azim.mean(), self.c_elev.mean())

        self.c_x_v = np.gradient(self.c_x) / self.dt
        self.c_y_v = np.gradient(self.c_y) / self.dt


DEFAULT_SITE_CONFIG = {'time_UTC' : 0,
                       'latitude' : -23.5,
                      'longitude' : -67.5,
                       'altitude' : 5e3,
                         'region' : 'chajnantor',
                          'epoch' : ttime.time(),
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

        self.weather = weathergen.generate(region=self.region, t=self.time_UTC)



DEFAULT_LAM_CONFIG = {'min_depth' : 500,
                      'max_depth' : 2500,
                       'n_layers' : 5,
                       'min_beam_res' : 2,
                       }

class LAM():
    
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


        # AM stuff, which takes us from physical atmosphere to detector powers

        #self.am = np.load('am.npy',allow_pickle=True)[()]

            
        self.am = np.load(path.join(path.abspath(path.dirname(__file__)), 'am.npy'),allow_pickle=True)[()]

        self.array.am_passbands  = sp.interpolate.interp1d(self.array.nu, self.array.passbands, bounds_error=False, fill_value=0, kind='cubic')(1e9*self.am['freq'])
        self.array.am_passbands /= self.array.am_passbands.sum(axis=1)[:,None]

        self.depths = np.linspace(self.min_depth, self.max_depth, self.n_layers)
        self.thicks = np.gradient(self.depths)

        self.waists = self.array.get_beam_waist(self.depths[:,None], self.array.primary_size, self.array.nu[None,:])

        self.azim, self.elev = tools.from_xy(self.array.x[:,None], self.array.y[:,None], self.plan.c_azim, self.plan.c_elev)
        
        self.unix = self.site.epoch + self.plan.time

        self.coordinator  = tools.coordinator(lon=self.site.longitude, lat=self.site.latitude)

        self.ra, self.dec = self.coordinator.transform(self.unix, self.azim, self.elev, in_frame='az_el',  out_frame='ra_dec')    
        self.l, self.b    = self.coordinator.transform(self.unix, self.ra,   self.dec,  in_frame='ra_dec', out_frame='l_b')

        self.angular_waists = self.waists / self.depths[:,None]

        self.min_ang_res = self.angular_waists / self.min_beam_res

        self.heights = self.depths[:,None] * np.sin(self.plan.c_elev[None,:])
        self.wvmd = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['water_density'])
        self.temp = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['temperature'])
        self.rel_scaling   = self.wvmd * self.temp * self.thicks[:, None]
        self.norm_scaling  = np.sqrt(np.square(self.rel_scaling) / np.square(self.rel_scaling).sum(axis=0))
        self.layer_scaling = np.exp(0.5 * self.site.weather['lbsd']) * self.norm_scaling 

        self.theta_x = self.array.x[:, None] + self.plan.c_x[None, :] 
        self.theta_y = self.array.y[:, None] + self.plan.c_y[None, :]

        self.theta_z = self.theta_x + 1j * self.theta_y

        self.w_e = np.interp(self.depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_east'])
        self.w_n = np.interp(self.depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_north'])

        self.w_b = np.arctan2(self.w_e, self.w_n)
        self.w_s = np.sqrt(np.square(self.w_e) + np.square(self.w_n))

        self.w_v_x = (+ self.w_e * np.cos(self.plan.c_azim[None,:]) - self.w_n * np.sin(self.plan.c_azim[None,:])) / self.depths[:,None]
        self.w_v_y = (- self.w_e * np.sin(self.plan.c_azim[None,:]) + self.w_n * np.cos(self.plan.c_azim[None,:])) / self.depths[:,None] * np.sin(self.plan.c_elev[None,:])
        
        ### These are empty lists we need to fill with chunky parameters (they won't fit together!) for each layer. 
        self.para, self.orth, self.P, self.O, self.X, self.Y = [], [], [], [], [], []
        self.n_para, self.n_orth, self.lay_ang_res, self.genz, self.AR_samples = [], [], [], [], []

        #self.rel_theta_z = self.theta_x + np.cumsum(self.w_v_x * self.plan.dt) + 1j# * (self.theta_y + np.cumsum(self.w_v_y * self.plan.dt))

        self.rel_c_x = self.plan.c_x[None,:] + np.cumsum(self.w_v_x * self.plan.dt, axis=1) 
        self.rel_c_y = self.plan.c_y[None,:] + np.cumsum(self.w_v_y * self.plan.dt, axis=1) 
        
        self.p   = np.zeros((self.n_layers, *self.theta_x.shape))
        self.o   = np.zeros((self.n_layers, *self.theta_x.shape))

        self.MARA = []
        self.outer_scale = 5e2
        self.ang_outer_scale = self.outer_scale / self.depths
        
        self.theta_edge_z = []

        radius_sample_prop = 1.5
        beam_tol = 1e-1

        max_layer_beam_radii = 0.5 * self.angular_waists.max(axis=1)

        self.padding = (radius_sample_prop + beam_tol) * max_layer_beam_radii
        
        for i_l, depth in enumerate(self.depths):

            rel_c  = np.c_[self.rel_c_x[i_l], self.rel_c_y[i_l]]
            rel_c += 1e-12 * np.random.standard_normal(size=rel_c.shape)

            hull = sp.spatial.qhull.ConvexHull(rel_c)
            h_x, h_y = hull.points[hull.vertices].T; h_z = h_x + 1j * h_y
            layer_hull_theta_z = h_z * (np.abs(h_z) + self.padding[i_l]) / np.abs(h_z)

            self.MARA.append(tools.get_MARA(layer_hull_theta_z.ravel()))

            rel_theta_x = self.array.x[:, None] + self.rel_c_x[i_l][None, :]
            rel_theta_y = self.array.y[:, None] + self.rel_c_y[i_l][None, :]
            
            zop = (rel_theta_x + 1j * rel_theta_y) * np.exp(1j*self.MARA[-1]) 
            self.p[i_l], self.o[i_l] = np.real(zop), np.imag(zop)

            res = self.min_ang_res[i_l].min()
            #lay_ang_res = np.minimum(self.min_ang_res[i_l].min(), 2 * orth_radius / (n_orth_min - 1))
            #lay_ang_res = np.maximum(lay_ang_res, 2 * orth_radius / (n_orth_max - 1))

            para_ = np.arange(self.p[i_l].min() - self.padding[i_l] - res, self.p[i_l].max() + self.padding[i_l] + res, res)
            orth_ = np.arange(self.o[i_l].min() - self.padding[i_l] - res, self.o[i_l].max() + self.padding[i_l] + res, res)

            n_para, n_orth = len(para_), len(orth_)
            self.lay_ang_res.append(res)

            # an efficient way to compute the minimal observing area that we need to generate
            #self.theta_edge_z.append(layer_hull_theta_z)

            #RZ = layer_hull_theta_z * np.exp(1j*self.MARA[-1])
            
            #para_min, para_max = np.real(RZ).min(), np.real(RZ).max()
            #orth_min, orth_max = np.imag(RZ).min(), np.imag(RZ).max()
            
            #para_center, orth_center = (para_min + para_max)/2, (orth_min + orth_max)/2
            #para_radius, orth_radius = (para_max - para_min)/2, (orth_max - orth_min)/2
    
            #n_orth_min = 64
            #n_orth_max = 1024

            
        


            #lay_ang_res = np.minimum(self.min_ang_res[i_l].min(), 2 * orth_radius / (n_orth_min - 1))
            #lay_ang_res = np.maximum(lay_ang_res, 2 * orth_radius / (n_orth_max - 1))

            
            
         
            self.PARA_SPACING = np.gradient(para_).mean()
            self.para.append(para_), self.orth.append(orth_)
            self.n_para.append(len(para_)), self.n_orth.append(len(orth_))
        
            ORTH_, PARA_ = np.meshgrid(orth_, para_)
            
            self.genz.append(np.exp(-1j*self.MARA[-1]) * (PARA_[0] + 1j*ORTH_[0] - res) )
            XYZ = np.exp(-1j*self.MARA[-1]) * (PARA_ + 1j*ORTH_) 
            
            self.X.append(np.real(XYZ)), self.Y.append(np.imag(XYZ))
            #self.O.append(ORTH_), self.P.append(PARA_)
            
            del rel_theta_x, rel_theta_y, zop

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

        

        self.data_type = np.float32



        
        with tqdm(total=len(self.depths),desc='Computing weights') as prog:
            for i_l, (depth, LX, LY, AR, GZ) in enumerate(zip(self.depths,self.X,self.Y,self.AR_samples,self.genz)):
                
                cov_args  = (self.outer_scale / depth, 5/6)
                
                self.prec.append(la.inv(tools.make_2d_covariance_matrix(tools.matern,cov_args,LX[AR],LY[AR])))

                self.cgen.append(tools.make_2d_covariance_matrix(tools.matern,cov_args,np.real(GZ),np.imag(GZ)))
                
                self.csam.append(tools.make_2d_covariance_matrix(tools.matern,cov_args,np.real(GZ),np.imag(GZ),LX[AR],LY[AR],auto=False)) 
                
                self.A.append(np.matmul(self.csam[i_l],self.prec[i_l]).astype(self.data_type)) 
                self.B.append(tools.msqrt(self.cgen[i_l]-np.matmul(self.A[i_l],self.csam[i_l].T)).astype(self.data_type))
                
                prog.update(1)

        if verbose:
            print('\n # | depth (m) | beam (m) | beam (\') | sim (m) | sim (\') | rms (mg/m2) | n_cov | orth | para | h2o (mg/m3) | temp (K) | ws (m/s) | wb (deg) |')
            
            for i_l, depth in enumerate(self.depths):
                
                row_string  = f'{i_l+1:2} | {depth:9.01f} | {self.waists[i_l].min():8.02f} | {60*np.degrees(self.angular_waists[i_l].min()):8.02f} | '
                row_string += f'{depth*self.lay_ang_res[i_l]:7.02f} | {60*np.degrees(self.lay_ang_res[i_l]):7.02f} | '
                row_string += f'{1e3*self.layer_scaling[i_l].mean():11.02f} | {len(self.AR_samples[i_l][0]):5} | {self.n_orth[i_l]:4} | '
                row_string += f'{self.n_para[i_l]:4} | {1e3*self.wvmd[i_l].mean():11.02f} | {self.temp[i_l].mean():8.02f} | '
                row_string += f'{self.w_s[i_l].mean():8.02f} | {np.degrees(self.w_b[i_l].mean()+np.pi):8.02f} |'
                print(row_string)

        


    def atmosphere_timestep(self,i): # iterate the i-th layer of atmosphere by one step
        
        self.vals[i] = np.r_[(np.matmul(self.A[i],self.vals[i][self.AR_samples[i]])
                            + np.matmul(self.B[i],np.random.standard_normal(self.B[i].shape[0]).astype(self.data_type)))[None,:],self.vals[i][:-1]]

    def initialize_atmosphere(self,blurred=False):

        self.vals = [np.zeros(lx.shape, dtype=self.data_type) for lx in self.X]
        n_init_   = [n_para for n_para in self.n_para]
        n_ts_     = [n_para for n_para in self.n_para]
        tot_n_init, tot_n_ts = np.sum(n_init_), np.sum(n_ts_)
        #self.gen_data = [np.zeros((n_ts,v.shape[1])) for n_ts,v in zip(n_ts_,self.lay_v_)]

        with tqdm(total=tot_n_init,desc='Generating layers') as prog:
            for i, n_init in enumerate(n_init_):
                for i_init in range(n_init):
                    
                    self.atmosphere_timestep(i)
                    prog.update(1)
                
        
    def simulate_atmosphere(self, do_atmosphere=True, verbose=False):
        
        self.sim_start = ttime.time()
        self.initialize_atmosphere()
        
        self.rel_flucs = np.zeros(self.o.shape, dtype=self.data_type)    
            
        multichromatic_beams = False

        from scipy.interpolate import RegularGridInterpolator as rgi

        
        with tqdm(total=self.n_layers, desc='Sampling layers') as prog:

            for i_d, d in enumerate(self.depths):

                # Compute the filtered line-of-sight pwv corresponding to each layer
                
                if multichromatic_beams:
                
                    filtered_vals   = np.zeros((len(self.array.nu), *self.vals[i_d].shape), dtype=self.data_type)
                    angular_res = self.lay_ang_res[i_d]

                    for i_f, f in enumerate(self.array.nu):

                        angular_waist = self.angular_waists[i_d, i_f]

                        self.F = self.array.make_filter(angular_waist, angular_res, self.array.beam_func)
                        u, s, v = self.array.separate_filter(self.F)

                        filtered_vals[i_f] = self.array.separably_filter(self.vals[i_d], u, s, v).astype(self.data_type)
                        
                else:
                    
                    angular_res   = self.lay_ang_res[i_d]
                    angular_waist = self.angular_waists[i_d].mean()

                    self.F = self.array.make_filter(angular_waist, angular_res, self.array.beam_func)

                    filtered_vals = self.array.separably_filter(self.vals[i_d], self.F).astype(self.data_type)
                    self.rel_flucs[i_d] = rgi((self.para[i_d], self.orth[i_d]), filtered_vals)((self.p[i_d], self.o[i_d]))

                prog.update(1)

        # Convert PWV fluctuations to detector powers

        self.epwv = self.site.weather['pwv'] + (self.rel_flucs * self.layer_scaling[:,None,:]).sum(axis=0)

        self.atm_power_data = np.zeros(self.epwv.shape)

        with tqdm(total=len(self.array.ubands), desc='Integrating spectra') as prog:
            for b in self.array.ubands:

                bm = self.array.bands == b

                ba_am_trj = (self.am['trj'] * self.array.am_passbands[bm].mean(axis=0)[None,None,None,:]).sum(axis=-1)

                BA_TRJ_RGI = sp.interpolate.RegularGridInterpolator((self.am['zpwv'], self.am['temp'], np.radians(self.am['elev'])), ba_am_trj)

                self.atm_power_data[bm] = BA_TRJ_RGI((self.epwv[bm], self.temp[0].mean(), self.elev[bm]))

                prog.update(1)



