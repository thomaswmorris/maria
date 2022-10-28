

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

from datetime import datetime




# how do we do the bands? this is a great question. 
# because all practical telescope instrumentation assume a constant band




# there are a few ways to define an array. 
# first, we need to determine the bands:

# the AUTO method requires you to pass:
#
# 'bands' is a list of bands genuses in Hz (e.g. [90e9, 150e9, 220e9])
# 'band_widths' has the same shape and determines the FWHM of the band
# 'dets_per_band' determines how many detectors there will be per band

# the MANUAL method requires you to pass:
#
# 
#
#
sites = weathergen.sites

def is_isoformat(x):
    try: datetime.fromisoformat(x); return True
    except: return False

class PointingError(Exception):
    pass


# this is the default array_config, which will be instantiated if another array_config is not passed to maria.Array()

DEFAULT_ARRAY_CONFIG = {        'site' : 'chajnantor',
                               'bands' : [(27e9, 10e9, 50), (39e9, 10e9, 50), (93e9, 10e9, 50), (145e9, 10e9, 50), (225e9, 10e9, 50), (280e9, 10e9, 50)],     # [Hz]  band centers
                            'geometry' : 'hex',     # [.]   type of detector distribution
                       'field_of_view' : 0.5,       # [deg] maximum det separation
                        'primary_size' : 12,         # [m]   size of the primary mirror
                       'band_grouping' : 'randomized', 
                    }

class Array():

    def __init__(self, config=DEFAULT_ARRAY_CONFIG, verbose=False):

        self.config = config

        for k, v in config.items(): setattr(self, k, v)

        # these are the minimal 
        #required_args_auto = ['bands', 'field_of_view', 'geometry']
        #if np.isin(required_args_auto, list(config.keys())).all():

        self.field_of_view = self.config['field_of_view']
        self.primary_size  = self.config['primary_size']
       
        self.bands, self.bandwidths, self.n_det = np.empty(0), np.empty(0), 0
        for nu_0, nu_w, n in self.config['bands']:
            self.bands, self.bandwidths = np.r_[self.bands, np.repeat(nu_0, n)], np.r_[self.bandwidths, np.repeat(nu_w, n)]
            self.n_det += n

        self.offsets  = tools.make_array(self.config['geometry'], self.field_of_view, self.n_det)
        self.offsets *= np.pi / 180 # put these in radians
        
        # compute detector offsets
        
        self.hull = sp.spatial.qhull.ConvexHull(self.offsets)

        # scramble up the locations of the bands 
        if self.config['band_grouping'] == 'random':
            random_index = np.random.choice(np.arange(self.n_det),self.n_det,replace=False)
            self.offsets = self.offsets[random_index]

        self.offset_x, self.offset_y = self.offsets.T
        self.r, self.p = np.sqrt(np.square(self.offsets).sum(axis=1)), np.arctan2(*self.offsets.T)

        self.ubands = np.unique(self.bands)
        self.nu = np.arange(0, 1e12, 1e9)
        
        self.passbands  = np.c_[[tools.get_passband(self.nu, nu_0, nu_w, order=16) for nu_0, nu_w in zip(self.bands, self.bandwidths)]]

        nu_mask = (self.passbands > 1e-4).any(axis=0)
        self.nu, self.passbands = self.nu[nu_mask], self.passbands[:, nu_mask]

        self.passbands /= self.passbands.sum(axis=1)[:,None]

        # compute beams
        self.optical_model = 'diff_lim'
        if self.optical_model == 'diff_lim':

            self.get_beam_waist   = lambda z, w_0, f : w_0 * np.sqrt(1 + np.square(2.998e8 * z) / np.square(f * np.pi * np.square(w_0)))
            self.get_beam_profile = lambda r, r_fwhm : np.exp(np.log(0.5)*np.abs(r/r_fwhm)**8)
            self.beam_func = self.get_beam_profile

        # position detectors:
        self.site = self.config['site']
        self.lat, self.lon, self.alt = sites.loc[sites.tag == self.site, ['lat', 'lon', 'alt']].values[0]

        self.coordinator  = tools.coordinator(lat=self.lat, lon=self.lon)

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


# the 'site' inherits from the weathergen site  

DEFAULT_PLAN_CONFIG = {    'start_time' : '2022-07-01T08:00:00',
                             'end_time' : '2022-07-01T08:10:00',
                         'scan_pattern' : 'daisy',      # [.]   the type of scan strategy (SS)
                         'scan_options' : {'k' : 3.1416}, # 
                         'coord_center' : (45, 45),
                          'coord_throw' : (2, 2),
                          'coord_frame' : 'azel',
                          'scan_period' : 120,        # [s]   how often the scan pattern repeats
                          'sample_rate' : 20,        # [Hz]  how fast to sample
                      }

class Plan():

    '''
    'back_and_forth' : back-and-forth        
    'daisy'          : lissajous daisy
    '''

    def __init__(self, config=DEFAULT_PLAN_CONFIG):

        self.config = config
        self.put(config)

    def put(self, config, verbose=False):

        for key, val in config.items(): 
            setattr(self, key, val)
            if verbose: print(f'set {key} to {val}')

        self.start_time = tools.datetime_handler(self.start_time)
        self.end_time   = tools.datetime_handler(self.end_time)

        self.compute()

    def compute(self):

        self.dt = 1 / self.sample_rate

        self.t_min = self.start_time.timestamp()
        self.t_max = self.end_time.timestamp()

        self.unix   = np.arange(self.t_min, self.t_max, self.dt)  
        self.coords = tools.get_pointing(self.unix, self.scan_period, self.coord_center, self.coord_throw, self.scan_pattern, self.scan_options)


DEFAULT_LAM_CONFIG = {'min_depth' : 500,
                      'max_depth' : 5000,
                       'n_layers' : 3,
                       'min_beam_res' : 8,
                       }

class LAM():
    
    def __init__(self, array, plan, config=DEFAULT_LAM_CONFIG, **kwargs):

        self.array, self.plan = array, plan
        self.config = {}
        
        for key, val in config.items(): 
            self.config[key] = val
            setattr(self, key, val)

        self.initialize(**kwargs)

    def initialize(self, verbose):


        #### COMPUTE POINTING ####

        if self.plan.coord_frame == 'az_el': 

            self.c_az, self.c_el  = np.radians(self.plan.coords[0]), np.radians(self.plan.coords[1])
            self.c_ra, self.c_dec = self.array.coordinator.transform(self.plan.unix, self.c_az.copy(), self.c_el.copy(), in_frame='az_el', out_frame='ra_dec') 

        if self.plan.coord_frame == 'ra_dec': 
            
            self.c_ra, self.c_dec = np.radians(self.plan.coords[0]), np.radians(self.plan.coords[1])
            self.c_az, self.c_el  = self.array.coordinator.transform(self.plan.unix, self.c_ra.copy(), self.c_dec.copy(), in_frame='ra_dec', out_frame='az_el') 


        self.c_x, self.c_y = tools.to_xy(self.c_az, self.c_el, self.c_az.mean(), self.c_el.mean())

        self.X = self.array.offset_x[:, None] + self.c_x[None, :] 
        self.Y = self.array.offset_y[:, None] + self.c_y[None, :]

        self.AZ, self.EL = tools.from_xy(self.X, self.Y, self.c_az.mean(), self.c_el.mean()) # get the 

        self.az_vel = np.gradient(self.c_az)   / np.gradient(self.plan.unix)
        self.az_acc = np.gradient(self.az_vel) / np.gradient(self.plan.unix)

        self.el_vel = np.gradient(self.c_el)   / np.gradient(self.plan.unix)
        self.el_acc = np.gradient(self.el_vel) / np.gradient(self.plan.unix)

        self.azim, self.elev = tools.from_xy(self.array.offset_x[:,None], self.array.offset_y[:,None], self.c_az, self.c_el)

        if self.elev.min() < np.radians(20):
            warnings.warn(f'Some detectors come within 20 degrees of the horizon, atmospheric model may be inaccurate (el_min = {np.degrees(self.elev.min()):.01f}°)')
        if self.elev.min() <= 0:
            raise PointingError(f'Some detectors are pointing below the horizon! (el_min = {np.degrees(self.elev.min()):.01f}°)')

        #### COMPUTE SPECTRA ####

        self.am = np.load(path.join(path.abspath(path.dirname(__file__)), 'am.npy'), allow_pickle=True)[()]

        self.array.am_passbands  = sp.interpolate.interp1d(self.array.nu, self.array.passbands, bounds_error=False, fill_value=0, kind='cubic')(1e9*self.am['freq'])
        self.array.am_passbands /= self.array.am_passbands.sum(axis=1)[:,None]

        #### COMPUTE ATMOSPHERIC LAYERS ####

        self.depths = np.linspace(self.min_depth, self.max_depth, self.n_layers)
        self.thicks = np.gradient(self.depths)

        self.waists = self.array.get_beam_waist(self.depths[:,None], self.array.primary_size, self.array.nu[None,:])

        self.angular_waists = self.waists / self.depths[:,None]

        self.min_ang_res = self.angular_waists / self.min_beam_res

        self.heights = self.depths[:,None,None] * np.sin(self.EL)[None]

        #### GENERATE WEATHER ####
        self.weather = weathergen.generate(site=self.array.site, time=np.arange(self.plan.t_min - 60, self.plan.t_max + 60, 60))
        for attr in ['abs_hum', 'air_temp', 'pressure', 'wind_north', 'wind_east']:

            #setattr(self, attr, sp.interpolate.RegularGridInterpolator((self.weather.time, self.weather.height - self.array.alt), \
            #                                                            getattr(self.weather, attr).T)((self.plan.unix, self.heights)))
            setattr(self, attr, sp.interpolate.interp1d((self.weather.height - self.array.alt), getattr(self.weather, attr).mean(axis=1))(self.heights))
        
        #self.wvmd = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['water_density'])
        #self.temp = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['temperature'])
        self.relative_scaling   = self.abs_hum * self.air_temp * self.thicks[:, None, None]
        self.layer_scaling      = np.sqrt(np.square(self.relative_scaling) / np.square(self.relative_scaling).sum(axis=0))


        

        #self.theta_z = self.THETA_X

        #self.w_e = np.interp(self.depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_east'])
        #self.w_n = np.interp(self.depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_north'])

        self.wind_bearing = np.arctan2(self.wind_east, self.wind_north)
        self.wind_speed = np.sqrt(np.square(self.wind_east) + np.square(self.wind_north))

        self.AWV_X = (+ self.wind_east * np.cos(self.AZ[None]) - self.wind_north * np.sin(self.AZ[None])) / self.depths[:,None,None]
        self.AWV_Y = (- self.wind_east * np.sin(self.AZ[None]) + self.wind_north * np.cos(self.AZ[None])) / self.depths[:,None,None] * np.sin(self.EL[None])
        
        self.REL_X = self.X[None] + np.cumsum(self.AWV_X * self.plan.dt, axis=-1)
        self.REL_Y = self.Y[None] + np.cumsum(self.AWV_Y * self.plan.dt, axis=-1) 

        self.rel_c_x, self.rel_c_y = self.REL_X.mean(axis=1), self.REL_Y.mean(axis=1)

        ### These are empty lists we need to fill with chunky parameters (they won't fit together!) for each layer. 
        self.para, self.orth, self.P, self.O, self.X, self.Y = [], [], [], [], [], []
        self.n_para, self.n_orth, self.lay_ang_res, self.genz, self.AR_samples = [], [], [], [], []

        #self.rel_theta_z = self.theta_x + np.cumsum(self.w_v_x * self.plan.dt) + 1j# * (self.theta_y + np.cumsum(self.w_v_y * self.plan.dt))
        
        self.p = np.zeros((self.REL_X.shape))
        self.o = np.zeros((self.REL_X.shape))

        self.MARA = []
        self.outer_scale = 5e2
        self.ang_outer_scale = self.outer_scale / self.depths
        
        self.theta_edge_z = []

        radius_sample_prop = 1.5
        beam_tol = 1e-2

        max_layer_beam_radii = 0.5 * self.angular_waists.max(axis=1)

        self.padding = (radius_sample_prop + beam_tol) * max_layer_beam_radii
        
        for i_l, depth in enumerate(self.depths):

            rel_c  = np.c_[self.rel_c_x[i_l], self.rel_c_y[i_l]]
            rel_c += 1e-12 * np.random.standard_normal(size=rel_c.shape)

            hull = sp.spatial.qhull.ConvexHull(rel_c)
            h_x, h_y = hull.points[hull.vertices].T; h_z = h_x + 1j * h_y
            layer_hull_theta_z = h_z * (np.abs(h_z) + self.padding[i_l]) / np.abs(h_z)

            self.MARA.append(tools.get_MARA(layer_hull_theta_z.ravel()))

            rel_theta_x = self.array.offset_x[:, None] + self.rel_c_x[i_l][None, :]
            rel_theta_y = self.array.offset_y[:, None] + self.rel_c_y[i_l][None, :]
            
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
            print('\n # | depth (m) | beam (m) | beam (\') | sim (m) | sim (\') | rms (mg/m2) | n_cov | orth | para | h2o (g/m3) | temp (K) | ws (m/s) | wb (deg) |')
            
            for i_l, depth in enumerate(self.depths):
                
                row_string  = f'{i_l+1:2} | {depth:9.01f} | {self.waists[i_l].min():8.02f} | {60*np.degrees(self.angular_waists[i_l].min()):8.02f} | '
                row_string += f'{depth*self.lay_ang_res[i_l]:7.02f} | {60*np.degrees(self.lay_ang_res[i_l]):7.02f} | '
                row_string += f'{1e3*self.layer_scaling[i_l].mean():11.02f} | {len(self.AR_samples[i_l][0]):5} | {self.n_orth[i_l]:4} | '
                row_string += f'{self.n_para[i_l]:4} | {1e3*self.abs_hum[i_l].mean():11.02f} | {self.air_temp[i_l].mean():8.02f} | '
                row_string += f'{self.wind_speed[i_l].mean():8.02f} | {np.degrees(self.wind_bearing[i_l].mean()+np.pi):8.02f} |'
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

        self.epwv = (self.rel_flucs * self.layer_scaling).sum(axis=0)

        self.epwv *= 5e1 / self.epwv.std()
        self.epwv += self.weather.pwv.mean()

        self.atm_power = np.zeros(self.epwv.shape)

        with tqdm(total=len(self.array.ubands), desc='Integrating spectra') as prog:
            for b in self.array.ubands:

                bm = self.array.bands == b

                ba_am_trj = (self.am['trj'] * self.array.am_passbands[bm].mean(axis=0)[None,None,None,:]).sum(axis=-1)

                BA_TRJ_RGI = sp.interpolate.RegularGridInterpolator((self.am['zpwv'], self.am['temp'], np.radians(self.am['elev'])), ba_am_trj)

                self.atm_power[bm] = BA_TRJ_RGI((self.epwv[bm], self.air_temp[0].mean(), self.elev[bm]))

                prog.update(1)



