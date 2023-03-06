

import numpy as np
import scipy as sp
import pandas as pd
import os
import h5py

from tqdm import tqdm

import warnings

from scipy import signal, spatial
from numpy import linalg as la

from importlib import resources
import time as ttime
from . import utils
import weathergen
from os import path

from datetime import datetime

sites = weathergen.sites

def is_isoformat(x):
    try: datetime.fromisoformat(x); return True
    except: return False

class PointingError(Exception):
    pass


# how do we do the bands? this is a great question. 
# because all practical telescope instrumentation assume a constant band


base, this_filename = os.path.split(__file__)

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

DEFAULT_LAM_CONFIG = {'min_depth' : 500,
                      'max_depth' : 5000,
                       'n_layers' : 3,
                       'min_beam_res' : 8,
                       }

class LinearAngularModel():
    
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

        self.c_x, self.c_y = utils.to_xy(self.c_az, self.c_el, self.c_az.mean(), self.c_el.mean())

        self.X = self.array.offset_x[:, None] + self.c_x[None, :] 
        self.Y = self.array.offset_y[:, None] + self.c_y[None, :]

        self.AZ, self.EL = utils.from_xy(self.X, self.Y, self.c_az.mean(), self.c_el.mean()) # get the 

        self.az_vel = np.gradient(self.c_az)   / np.gradient(self.plan.unix)
        self.az_acc = np.gradient(self.az_vel) / np.gradient(self.plan.unix)

        self.el_vel = np.gradient(self.c_el)   / np.gradient(self.plan.unix)
        self.el_acc = np.gradient(self.el_vel) / np.gradient(self.plan.unix)

        self.azim, self.elev = utils.from_xy(self.array.offset_x[:,None], self.array.offset_y[:,None], self.c_az, self.c_el)

        if self.elev.min() < np.radians(20):
            warnings.warn(f'Some detectors come within 20 degrees of the horizon, atmospheric model may be inaccurate (el_min = {np.degrees(self.elev.min()):.01f}°)')
        if self.elev.min() <= 0:
            raise PointingError(f'Some detectors are pointing below the horizon! (el_min = {np.degrees(self.elev.min()):.01f}°)')

        #### COMPUTE SPECTRA ####

        class AM():
            def __init__(self):
                '''
                A dummy class to hold AM spectra as attributes
                '''
                pass

        self.am = AM()
        self.am.filepath = f'{base}/am/{self.array.site}.h5'
        with h5py.File(self.am.filepath, 'r') as f:
            self.am.nu = f['nu'][:] # frequency axis of the spectrum, in GHz
            self.am.tcwv = f['tcwv'][:] # total column water vapor, in mm
            self.am.elev = f['elev'][:] # elevation, in degrees
            self.am.t_rj = f['t_rj'][:] # Rayleigh-Jeans temperature, in Kelvin

        self.array.am_passbands  = sp.interpolate.interp1d(self.array.nu, self.array.passbands, bounds_error=False, fill_value=0, kind='cubic')(1e9*self.am.nu)
        self.array.am_passbands /= self.array.am_passbands.sum(axis=1)[:,None]

        #### COMPUTE ATMOSPHERIC LAYERS ####

        self.layer_depths = np.linspace(self.min_depth, self.max_depth, self.n_layers)
        self.layer_thicks = np.gradient(self.layer_depths)

        self.waists = self.array.get_beam_waist(self.layer_depths[:,None], self.array.primary_size, self.array.nu[None,:])

        self.angular_waists = self.waists / self.layer_depths[:,None]

        self.min_ang_res = self.angular_waists / self.min_beam_res

        self.heights = self.layer_depths[:,None,None] * np.sin(self.EL)[None]

        #### GENERATE WEATHER ####
        self.weather = weathergen.generate(site=self.array.site, time=np.arange(self.plan.t_min - 60, self.plan.t_max + 60, 60))
        for attr in ['water_vapor', 'temperature', 'pressure', 'wind_north', 'wind_east']:

            #setattr(self, attr, sp.interpolate.RegularGridInterpolator((self.weather.time, self.weather.height - self.array.alt), \
            #                                                            getattr(self.weather, attr).T)((self.plan.unix, self.heights)))
            setattr(self, attr, sp.interpolate.interp1d((self.weather.height - self.array.altitude), getattr(self.weather, attr).mean(axis=1))(self.heights))
        
        #self.wvmd = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['water_density'])
        #self.temp = np.interp(self.heights, self.site.weather['height'] - self.site.altitude, self.site.weather['temperature'])
        self.relative_scaling   = self.water_vapor * self.temperature * self.layer_thicks[:, None, None]
        self.layer_scaling      = np.sqrt(np.square(self.relative_scaling) / np.square(self.relative_scaling).sum(axis=0))


        

        #self.theta_z = self.THETA_X

        #self.w_e = np.interp(self.layer_depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_east'])
        #self.w_n = np.interp(self.layer_depths[:,None] * np.sin(self.plan.c_elev)[None,:], self.site.weather['height'] - self.site.altitude, self.site.weather['wind_north'])

        self.wind_bearing = np.arctan2(self.wind_east, self.wind_north)
        self.wind_speed = np.sqrt(np.square(self.wind_east) + np.square(self.wind_north))

        self.AWV_X = (+ self.wind_east * np.cos(self.AZ[None]) - self.wind_north * np.sin(self.AZ[None])) / self.layer_depths[:,None,None]
        self.AWV_Y = (- self.wind_east * np.sin(self.AZ[None]) + self.wind_north * np.cos(self.AZ[None])) / self.layer_depths[:,None,None] * np.sin(self.EL[None])
        
        self.REL_X = self.X[None] + np.cumsum(self.AWV_X * self.plan.dt, axis=-1)
        self.REL_Y = self.Y[None] + np.cumsum(self.AWV_Y * self.plan.dt, axis=-1) 

        self.rel_c_x, self.rel_c_y = self.REL_X.mean(axis=1), self.REL_Y.mean(axis=1)

        ### These are empty lists we need to fill with chunky parameters (they won't fit together!) for each layer. 
        self.para, self.orth, self.P, self.O, self.X, self.Y = [], [], [], [], [], []
        self.n_para, self.n_orth, self.lay_ang_res, self.genz, self.AR_samples = [], [], [], [], []

        #self.rel_theta_z = self.theta_x + np.cumsum(self.w_v_x * self.plan.dt) + 1j# * (self.theta_y + np.cumsum(self.w_v_y * self.plan.dt))
        
        self.p = np.zeros((self.REL_X.shape))
        self.o = np.zeros((self.REL_X.shape))

        self.layer_rotation_angles = []
        self.outer_scale = 5e2
        self.ang_outer_scale = self.outer_scale / self.layer_depths
        
        self.theta_edge_z = []

        radius_sample_prop = 1.5
        beam_tol = 1e-2

        max_layer_beam_radii = 0.5 * self.angular_waists.max(axis=1)

        self.padding = (radius_sample_prop + beam_tol) * max_layer_beam_radii
        
        for i_l, depth in enumerate(self.layer_depths):

            rel_c  = np.c_[self.rel_c_x[i_l], self.rel_c_y[i_l]]
            rel_c += 1e-12 * np.random.standard_normal(size=rel_c.shape)

            hull = sp.spatial.ConvexHull(rel_c)
            h_x, h_y = hull.points[hull.vertices].T; h_z = h_x + 1j * h_y
            layer_hull_theta_z = h_z * (np.abs(h_z) + self.padding[i_l]) / np.abs(h_z)

            self.layer_rotation_angles.append(utils.get_minimal_bounding_rotation_angle(layer_hull_theta_z.ravel()))

            rel_theta_x = self.array.offset_x[:, None] + self.rel_c_x[i_l][None, :]
            rel_theta_y = self.array.offset_y[:, None] + self.rel_c_y[i_l][None, :]
            
            zop = (rel_theta_x + 1j * rel_theta_y) * np.exp(1j*self.layer_rotation_angles[-1]) 
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
            
            self.genz.append(np.exp(-1j*self.layer_rotation_angles[-1]) * (PARA_[0] + 1j*ORTH_[0] - res) )
            XYZ = np.exp(-1j*self.layer_rotation_angles[-1]) * (PARA_ + 1j*ORTH_) 
            
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



        
        with tqdm(total=len(self.layer_depths),desc='Computing weights') as prog:
            for i_l, (depth, LX, LY, AR, GZ) in enumerate(zip(self.layer_depths,self.X,self.Y,self.AR_samples,self.genz)):
                
                cov_args  = (self.outer_scale / depth, 5/6)
                
                self.prec.append(la.inv(utils.make_2d_covariance_matrix(utils.matern,cov_args,LX[AR],LY[AR])))

                self.cgen.append(utils.make_2d_covariance_matrix(utils.matern,cov_args,np.real(GZ),np.imag(GZ)))
                
                self.csam.append(utils.make_2d_covariance_matrix(utils.matern,cov_args,np.real(GZ),np.imag(GZ),LX[AR],LY[AR],auto=False)) 
                
                self.A.append(np.matmul(self.csam[i_l],self.prec[i_l]).astype(self.data_type)) 
                self.B.append(utils.msqrt(self.cgen[i_l]-np.matmul(self.A[i_l],self.csam[i_l].T)).astype(self.data_type))
                
                prog.update(1)

        if verbose:
            print('\n # | depth (m) | beam (m) | beam (\') | sim (m) | sim (\') | rms (mg/m2) | n_cov | orth | para | h2o (g/m3) | temp (K) | ws (m/s) | wb (deg) |')
            
            for i_l, depth in enumerate(self.layer_depths):
                
                row_string  = f'{i_l+1:2} | {depth:9.01f} | {self.waists[i_l].min():8.02f} | {60*np.degrees(self.angular_waists[i_l].min()):8.02f} | '
                row_string += f'{depth*self.lay_ang_res[i_l]:7.02f} | {60*np.degrees(self.lay_ang_res[i_l]):7.02f} | '
                row_string += f'{1e3*self.layer_scaling[i_l].mean():11.02f} | {len(self.AR_samples[i_l][0]):5} | {self.n_orth[i_l]:4} | '
                row_string += f'{self.n_para[i_l]:4} | {1e3*self.water_vapor[i_l].mean():11.02f} | {self.temperature[i_l].mean():8.02f} | '
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

            for i_d, d in enumerate(self.layer_depths):

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

        self.epwv *= 5e-2 / self.epwv.std()
        self.epwv += self.weather.column_water_vapor.mean()

        self.atm_power = np.zeros(self.epwv.shape)

        with tqdm(total=len(self.array.ubands), desc='Integrating spectra') as prog:
            for b in self.array.ubands:

                bm = self.array.bands == b

                ba_am_trj = (self.am.t_rj * self.array.am_passbands[bm].mean(axis=0)[None,None,:]).sum(axis=-1)

                BA_TRJ_RGI = sp.interpolate.RegularGridInterpolator((self.am.tcwv, np.radians(self.am.elev)), ba_am_trj)

                self.atm_power[bm] = BA_TRJ_RGI((self.epwv[bm], self.elev[bm]))

                prog.update(1)



def get_extrusion_products(time, azim, elev, 
                           baseline, 
                           field_of_view, 
                           min_frequency,
                           wind_velocity,
                           wind_direction, 
                           max_depth=3000, 
                           extrusion_step=1,
                           min_res=10, 
                           max_res=100):
    '''
    For the Kolomogorov-Taylor model. It works the same as one of those pasta extruding machines (https://youtu.be/TXtm_eNaIwQ). This function figures out the shape of the hole.
    
    The inputs are:
    
    azim, elev: time-ordered pointing
    '''
    
    # Within this function, we work in the frame of wind-relative pointing phi'. The elevation stays the same. We define: 
    # 
    # phi' -> phi - phi_w
    #
    # where phi_w is the wind bearing, i.e. the direction the wind comes from. This means that:
    #
    # For phi' = 0°  you are looking into the wind
    # For phi' = 90° the wind is going from left to right in your field of view.
    #
    # We further define the frame (x, y, z) = (r cos phi' sin theta, r sin phi' sin theta, r sin theta) 
    
    
    n_baseline = len(baseline)
    
    # this converts the $(x, y, z)$ vertices of a beam looking straight up to the $(x, y, z)$ vertices of 
    # the time-ordered pointing of the beam in the frame of the wind direction.
    elev_rotation_matrix  = sp.spatial.transform.Rotation.from_euler('y', np.pi/2 - elev).as_matrix()
    azim_rotation_matrix  = sp.spatial.transform.Rotation.from_euler('z', np.pi/2 - azim + wind_direction).as_matrix()
    total_rotation_matrix = np.matmul(azim_rotation_matrix, elev_rotation_matrix)

    # this defines the maximum size of the beam that we have to worry about 
    max_beam_radius      = _beam_sigma(max_depth, primary_size=primary_sizes, nu=5e9)
    max_boresight_offset = max_beam_radius + max_depth * field_of_view / 2

    # this returns a list of time-ordered vertices, which can used to construct a convex hull
    time_ordered_vertices_list = []
    for _min_radius, _max_radius, _baseline in zip(primary_sizes/2, max_boresight_offset, baseline):

        _rot_baseline = np.matmul(sp.spatial.transform.Rotation.from_euler('z', wind_direction).as_matrix(), _baseline)
        
        _vertices = np.c_[np.r_[[_.ravel() for _ in np.meshgrid([-_min_radius, _min_radius], [-_min_radius, _min_radius], [0])]], 
                          np.r_[[_.ravel() for _ in np.meshgrid([-_max_radius, _max_radius], [-_max_radius, _max_radius], [max_depth])]]]
        
        time_ordered_vertices_list.append(_rot_baseline[None] + np.swapaxes(np.matmul(total_rotation_matrix, _vertices), 1, -1).reshape(-1, 3))

    # these are the bounds in space that we have to worry about. it has shape (3, 2)
    extrusion_bounds = np.c_[np.min([np.min(tov, axis=0) for tov in time_ordered_vertices_list], axis=0),
                             np.max([np.max(tov, axis=0) for tov in time_ordered_vertices_list], axis=0)]

    # here we make the layers of the atmosphere given the bounds of $z$ and the specified resolution; we use 
    # regular layers to make interpolation more efficient later
    min_height = np.max([np.max(tov[:,2][tov[:,2] < np.median(tov[:,2])]) for tov in time_ordered_vertices_list])
    max_height = np.min([np.min(tov[:,2][tov[:,2] > np.median(tov[:,2])]) for tov in time_ordered_vertices_list])
    
    height_samples = np.linspace(min_height, max_height, 1024)
    dh = np.gradient(height_samples).mean()
    dheight_dindex = np.interp(height_samples, [min_height, max_height], [min_res, max_res])
    dindex_dheight = 1 / dheight_dindex
    n_layers = int(np.sum(dindex_dheight * dh))
    layer_heights = sp.interpolate.interp1d(np.cumsum(dindex_dheight * dh), 
                                            height_samples, 
                                            bounds_error=False, 
                                            fill_value='extrapolate')(1 + np.arange(n_layers))

    # here we define the cells through which turbulence will be extruded.
    layer_res = np.gradient(layer_heights)
    x_min, x_max = extrusion_bounds[0,0], extrusion_bounds[0,1]
    n_per_layer = ((x_max - x_min) / layer_res).astype(int)
    cell_x = np.concatenate([np.linspace(x_min, x_max, n) for i, (res, n) in enumerate(zip(layer_res, n_per_layer))])
    cell_z = np.concatenate([h * np.ones(n) for i, (h, n) in enumerate(zip(layer_heights, n_per_layer))])
    cell_res = np.concatenate([res * np.ones(n) for i, (res, n) in enumerate(zip(layer_res, n_per_layer))])
    
    extrusion_cells = np.c_[cell_x, cell_z]
    extrusion_shift = np.c_[cell_res, np.zeros(len(cell_z))]
    
    eps = 1e-6
    
    in_view = np.zeros(len(extrusion_cells)).astype(bool)
    for tov in time_ordered_vertices_list:
        
        baseline_in_view = np.zeros(len(extrusion_cells)).astype(bool)

        hull = sp.spatial.ConvexHull(tov[:,[0,2]]) # we only care about the $(x, z)$ dimensions here
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        
        for shift_factor in [-1, 0, +1]:
        
            baseline_in_view |= np.all((extrusion_cells + shift_factor * extrusion_shift) @ A.T + b.T < eps, axis=1)

        in_view |= baseline_in_view # if the cell is in view of any of the baselines, keep it!
        
        #plt.scatter(tov[:, 0], tov[:, 2], s=1e0)

    # here we define the extrusion axis
    y_min, y_max = extrusion_bounds[1,0], extrusion_bounds[1,1] + wind_velocity * time.ptp()
    extrustion_axis = np.arange(y_min, y_max, extrusion_step)
    
    return extrusion_cells[in_view].T, extrustion_axis, cell_res[in_view]
    
    





class KolmogorovTaylorModel():
    
    def __init__(self, 
                 time, 
                 azim, 
                 elev, 
                 baseline, 
                 field_of_view, 
                 min_frequency,
                 wind_velocity,
                 wind_direction, 
                 extrusion_step=1,
                 outer_scale=600,
                 min_res=10, 
                 max_res=100,
                 max_depth=3000
                ):
        
        
        self.time = time
        self.azim = azim
        self.elev = elev
        
        self.baseline       = baseline
        self.field_of_view  = field_of_view
        self.min_frequency  = min_frequency
        self.wind_direction = wind_direction
        self.wind_velocity  = wind_velocity
        self.extrusion_step = extrusion_step
        self.min_res = min_res
        self.max_res = max_res
        self.max_depth = max_depth
        
        self.outer_scale = outer_scale
        
        
        (self.cX, self.cZ), self.Y, self.cres = get_extrusion_products(self.time, self.azim, self.elev, 
                                                                       self.baseline, 
                                                                       self.field_of_view, 
                                                                       self.min_frequency,
                                                                       self.wind_velocity,
                                                                       self.wind_direction, 
                                                                       self.max_depth, 
                                                                       self.extrusion_step,
                                                                       self.min_res, 
                                                                       self.max_res)
        
        self.layer_heights, self.layer_index = np.unique(self.cZ, return_inverse=True)
        self.n_layers = len(self.layer_heights)

        self.dY, self.n_cells = np.gradient(self.Y).mean(), len(self.cX)
        self.n_baseline  = len(self.baseline)
        self.n_extrusion = len(self.Y)
        self.n_history   = int(2 * self.outer_scale / self.dY) + 1
        
        self.initialized = False
        
    def initialize(self):
        
        max_spacing = int(self.n_cells / 1.1)
        iter_samples = [np.random.choice(self.n_cells, int(self.n_cells / np.minimum(2**i, max_spacing)), replace=False) for i in range(self.n_history)]
        self.hist_iter_index = np.concatenate([i * np.ones(len(index), dtype=int) for i, index in enumerate(iter_samples)])
        self.cell_iter_index = np.concatenate(iter_samples).astype(int)
        
        
        Xi, Xj = self.cX, self.cX[self.cell_iter_index]
        Yi, Yj = np.zeros(self.n_cells), self.dY * (1 + self.hist_iter_index)
        Zi, Zj = self.cZ, self.cZ[self.cell_iter_index]

        Rii = np.sqrt(np.subtract.outer(Xi, Xi) ** 2
                    + np.subtract.outer(Yi, Yi) ** 2
                    + np.subtract.outer(Zi, Zi) ** 2)

        Rij = np.sqrt(np.subtract.outer(Xi, Xj) ** 2
                    + np.subtract.outer(Yi, Yj) ** 2
                    + np.subtract.outer(Zi, Zj) ** 2)

        Rjj = np.sqrt(np.subtract.outer(Xj, Xj) ** 2
                    + np.subtract.outer(Yj, Yj) ** 2
                    + np.subtract.outer(Zj, Zj) ** 2)
        
        alpha = 1e-3 

        # this is all very computationally expensive stuff (n^3 is rough!)
        self.Cii  = utils._approximate_normalized_matern(Rii, r0=self.outer_scale, nu=1/3, n_test_points=4096) + alpha ** 2 * np.eye(self.n_cells)
        self.Cij  = utils._approximate_normalized_matern(Rij, r0=self.outer_scale, nu=1/3, n_test_points=4096) 
        self.Cjj  = utils._approximate_normalized_matern(Rjj, r0=self.outer_scale, nu=1/3, n_test_points=4096) 
        self.A    = np.matmul(self.Cij, utils._fast_psd_inverse(self.Cjj))
        self.B, _ = sp.linalg.lapack.dpotrf(self.Cii - np.matmul(self.A, self.Cij.T))

        self.initialized = True
        
    def extrude(self):
        
        if not self.initialized:
            self.initialize()
            
        ARH = np.zeros((self.n_cells, self.n_history))
        self.cdata = np.zeros((self.n_cells, self.n_extrusion))

        def iterate_autoregression(ARH, n_iter):
            for i in range(n_iter):
                res = np.matmul(self.A, ARH[self.cell_iter_index, self.hist_iter_index]) + np.matmul(self.B, np.random.standard_normal(self.n_cells))
                ARH = np.c_[res, ARH[:,:-1]]
            return ARH

        ARH = iterate_autoregression(ARH, n_iter=2*self.n_history)

        for i in range(self.n_extrusion):
            ARH = iterate_autoregression(ARH, n_iter=1)
            self.cdata[:,i] = ARH[:,0]