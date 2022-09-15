# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import astropy as ap

from numpy import linalg as la
from scipy import cluster
from astropy import time, coordinates








# COORDINATE TRANSFORM UTILS

    
def validate_args(DICT, necessary_args):
    missing_args = []
    for arg in necessary_args: 
        if not arg in DICT:
            missing_args.append(arg)
    if not len(missing_args) == 0: 
        raise Exception(f'missing arguments {missing_args}')

def baf_pointing(time, period, az_center, az_throw, el_center, el_throw, plan_options={}):

    validate_args(plan_options, [])                                                        
    p = 2 * np.pi * time / period
    return az_center+az_throw*sp.signal.sawtooth(p, width=0.5), el_center+el_throw*sp.signal.sawtooth(p, width=0.5)

def box_pointing(time, period, az_center, az_throw, el_center, el_throw, plan_options={}):

    validate_args(plan_options, [])                                                        
    p = 2 * np.pi * time / period
    return az_center+az_throw*np.interp(p % (2*np.pi),np.linspace(0,2*np.pi,5),[-1,-1,+1,+1,-1]), el_center+el_throw*np.interp(p,np.linspace(0,2*np.pi,5),[-1,+1,+1,-1,-1])

def rose_pointing(time, period, az_center, az_throw, el_center, el_throw, plan_options={}): 

    validate_args(plan_options, ['k'])                                                        
    p = 2 * np.pi * time / period
    return az_center+az_throw*np.sin(plan_options['k']*p)*np.cos(p), el_center+el_throw*np.sin(plan_options['k']*p)*np.sin(p)

def lissajous_pointing(time, period, az_center, az_throw, el_center, el_throw, plan_options={}): 

    validate_args(plan_options, ['k_az', 'k_el'])                                                        
    p = 2 * np.pi * time / period
    return az_center+az_throw*np.sin(plan_options['k_az']*p), el_center+el_throw*np.sin(plan_options['k_el']*p)

def get_pointing(time, period, az_center, az_throw, el_center, el_throw, plan_type, plan_options):

    if plan_type == 'baf' :       plan_func = baf_pointing
    if plan_type == 'box' :       plan_func = box_pointing
    if plan_type == 'rose' :      plan_func = rose_pointing
    if plan_type == 'lissajous' : plan_func = lissajous_pointing

    return plan_func(time, period, az_center, az_throw, el_center, el_throw, plan_options)

class coordinator():

    # what three-dimensional rotation matrix takes (frame 1) to (frame 2) ? 
    # we can brute-force this for a few test points, and then use it to efficiently broadcast very big arrays

    def __init__(self, lon, lat):

        #if frame is None:
        #    raise ValueError('Please provide a frame')

        self.fid_p = np.radians(np.array([0,0,90]))
        self.fid_t = np.radians(np.array([90,89,89]))

        self.lc = ap.coordinates.EarthLocation.from_geodetic(lon=lon,lat=lat)

        self.fid_xyz = np.c_[np.sin(self.fid_p) * np.cos(self.fid_t), np.cos(self.fid_p) * np.cos(self.fid_t), np.sin(self.fid_t)] 

        # you are standing a the north pole looking toward lon = -90 (+x)
        # you are standing a the north pole looking toward lon = 0 (+y)
        # you are standing a the north pole looking up (+z)

    def transform(self, unix, phi, theta, in_frame, out_frame):
        
        epoch = unix.mean()
        ot    = ap.time.Time(epoch, format='unix')
        rad   = ap.units.rad
        
        if in_frame == 'az_el':  self.c = ap.coordinates.SkyCoord(az = self.fid_p * rad, alt = self.fid_t * rad, obstime = ot, frame = 'altaz',    location = self.lc)
        if in_frame == 'ra_dec': self.c = ap.coordinates.SkyCoord(ra = self.fid_p * rad, dec = self.fid_t * rad, obstime = ot, frame = 'icrs',     location = self.lc)
        if in_frame == 'l_b':    self.c = ap.coordinates.SkyCoord(l  = self.fid_p * rad, b   = self.fid_t * rad, obstime = ot, frame = 'galactic', location = self.lc)

        if out_frame == 'ra_dec': self._c = self.c.icrs;     self.rot_p, self.rot_t = self._c.ra.rad, self._c.dec.rad
        if out_frame == 'az_el':  self._c = self.c.altaz;    self.rot_p, self.rot_t = self._c.az.rad, self._c.alt.rad
        if out_frame == 'l_b':    self._c = self.c.galactic; self.rot_p, self.rot_t = self._c.l.rad,  self._c.b.rad
            
        self.rot_xyz = np.c_[np.sin(self.rot_p) * np.cos(self.rot_t), np.cos(self.rot_p) * np.cos(self.rot_t), np.sin(self.rot_t)] 

        # we want to find T in the equation [ T x1 = x2 ] for our test points (T is a rotation, with maybe a reflection)
        # A x = B

        self.R = la.lstsq(self.fid_xyz, self.rot_xyz, rcond=None)[0]

        if (in_frame, out_frame) == ('ra_dec', 'az_el'): phi -= (unix - epoch) * (2*np.pi/86163.0905)

        xyz = np.matmul(np.concatenate([(np.sin(phi) * np.cos(theta))[:,:,None], (np.cos(phi) * np.cos(theta))[:,:,None], np.sin(theta)[:,:,None]],axis=-1), self.R)

        _phi, _theta = np.arctan2(xyz[:,:,0], xyz[:,:,1]), np.arcsin(xyz[:,:,2])

        if (in_frame, out_frame) == ('az_el', 'ra_dec'): _phi += (unix - epoch) * (2*np.pi/86163.0905)

        return _phi % (2 * np.pi), _theta



# ================ ARRAY ================



def get_passband(nu, nu_0, nu_w, order=4):
    return np.exp(-np.abs(2*(nu-nu_0)/nu_w)**order)


def make_array(array_shape, max_fov, n_det):

    valid_array_types = ['flower', 'hex', 'square']
    
    if array_shape=='flower':
        phi = np.pi*(3.-np.sqrt(5.))  # golden angle in radians
        dzs = np.zeros(n_det).astype(complex)
        for i in range(n_det):
            dzs[i] = np.sqrt((i / (n_det - 1)) * 2 ) *np.exp(1j*phi*i)
        od = np.abs(np.subtract.outer(dzs,dzs))
        dzs *= max_fov / od.max()
        return np.c_[np.real(dzs), np.imag(dzs)]
    if array_shape=='hex':
        return make_hex(n_det,max_fov)
    if array_shape=='square':
        dxy_ = np.linspace(-max_fov,max_fov,int(np.ceil(np.sqrt(n_det))))/(2*np.sqrt(2))
        DX, DY = np.meshgrid(dxy_,dxy_)
        return np.c_[DX.ravel()[:n_det], DY.ravel()[:n_det]]
    
    raise ValueError('Please specify a valid array type. Valid array types are:\n'+
              '\n'.join(valid_array_types))
    
    
def make_hex(n,d):
    
    angles = np.linspace(0,2*np.pi,6+1)[1:] + np.pi/2
    zs     = np.array([0])
    layer  = 0
    while len(zs) < n:
        for angle in angles:
            for z in layer*np.exp(1j*angle) + np.arange(layer)*np.exp(1j*(angle+2*np.pi/3)):
                zs = np.append(zs,z)
        layer += 1
    zs -= zs.mean()
    zs *= .5 * d / np.abs(zs).max() 
                
    return np.c_[np.real(np.array(zs[:n])), np.imag(np.array(zs[:n]))]

# ================ STATS ================

msqrt  = lambda M : [np.matmul(u,np.diag(np.sqrt(s))) for u,s,vh in [la.svd(M)]][0]
matern = lambda r,r0,nu : 2**(1-nu)/sp.special.gamma(nu)*sp.special.kv(nu,r/r0+1e-10)*(r/r0+1e-10)**nu
    
gaussian_beam = lambda z, w0, l, n : np.sqrt(1/np.square(z) + np.square(l) / np.square(w0 * np.pi * n))

def get_MARA(z): # minimal-area rotation angle 

    H  = sp.spatial.ConvexHull(points=np.vstack([np.real(z).ravel(),np.imag(z).ravel()]).T)
    HZ = z.ravel()[H.vertices]

    HE = np.imag(HZ).max() - np.imag(HZ).min(); HO = 0
    #for z1,z2 in zip(HZ,np.roll(HZ,1)):
    for RHO in np.linspace(0,np.pi,1024+1)[1:]:

        #RHO = np.angle(z2-z1)
        RHZ = HZ * np.exp(1j*RHO)
        
        im_width = (np.imag(RHZ).max() - np.imag(RHZ).min())
        re_width = (np.real(RHZ).max() - np.real(RHZ).min())
        
        RHE = im_width #* re_width

        if RHE < HE and re_width > im_width: 
            HE = RHE; HO = RHO
            
    return HO

def smallest_max_error_sample(items, max_error=1e0):
        
    k = 1
    cluster_mids = np.sort(sp.cluster.vq.kmeans(items,k_or_guess=1)[0])
    while (np.abs(np.subtract.outer(items,cluster_mids)) / cluster_mids[None,:]).min(axis=1).max() > max_error:
        cluster_mids = np.sort(sp.cluster.vq.kmeans(items,k_or_guess=k)[0])
        k += 1

    which_cluster = np.abs(np.subtract.outer(items,cluster_mids)).argmin(axis=1)
    
    return cluster_mids, which_cluster

def make_beam_filter(side,window_func,args):
    
    beam_X, beam_Y = np.meshgrid(side,side)
    beam_R = np.sqrt(np.square(beam_X)+np.square(beam_Y))
    beam_W = window_func(beam_R,*args)
    
    return beam_W / beam_W.sum()


def make_2d_covariance_matrix(C,args,x0,y0,x1=None,y1=None,auto=True):
    if auto:
        n = len(x0); i,j = np.triu_indices(n,1)
        o = C(np.sqrt(np.square(x0[i] - x0[j]) + np.square(y0[i] - y0[j])),*args)
        c = np.empty((n,n)); c[i,j],c[j,i] = o,o
        c[np.eye(n).astype(bool)] = 1
    if not auto:
        n = len(x0); i,j = np.triu_indices(n,1)
        c = C(np.sqrt(np.square(np.subtract.outer(x0,x1))
                    + np.square(np.subtract.outer(y0,y1))),*args)
    return c



def get_sub_splits(time_,xvel_,durations=[]):

    # flag wherever the scan velocity changing direction (can be more general)
    flags  = np.r_[0,np.where(np.sign(xvel_[:-1]) != np.sign(xvel_[1:]))[0],len(xvel_)-1]
    splits = np.array([[s,e] for s,e in zip(flags,flags[1:])]).astype(int)

    # compiles sub-scans that cover the TOD
    sub_splits = splits.copy()
    dt = np.median(np.gradient(time_))
    for i,(s,e) in enumerate(splits):

        split_dur = time_[e] - time_[s]
        for pld in durations:
            if pld > split_dur:
                continue
            sub_n = 2 * int(split_dur / pld) + 1
            sub_s = np.linspace(s,e-int(pld/dt),sub_n).astype(int)
            for sub_s in np.linspace(s,e-int(np.ceil(pld/dt)),sub_n).astype(int):
                sub_splits = np.r_[sub_splits,np.array([sub_s,sub_s+int(pld/dt)])[None,:]]

    return sub_splits
    

    
def get_brightness_temperature(f_pb,pb,f_spec,spec):

    return sp.integrate.trapz(sp.interpolate.interp1d(f_spec, spec, axis=-1)(f_pb)*pb,f_pb,axis=-1) / sp.integrate.trapz(pb,f_pb)

# ================ POINTING ================

def to_xy(p, t, c_p, c_t):
    ground_X, ground_Y, ground_Z = np.sin(p-c_p)*np.cos(t), np.cos(p-c_p)*np.cos(t), np.sin(t)
    return np.arcsin(ground_X), np.arcsin(-np.real((ground_Y+1j*ground_Z)*np.exp(1j*(np.pi/2-c_t))))

def from_xy(dx, dy, c_p, c_t):
    ground_X, Y, Z = np.sin(dx+1e-16), -np.sin(dy+1e-16), np.cos(np.sqrt(dx**2+dy**2))
    gyz = (Y+1j*Z)*np.exp(-1j*(np.pi/2-c_t))
    ground_Y, ground_Z = np.real(gyz), np.imag(gyz)
    return (np.angle(ground_Y+1j*ground_X) + c_p) % (2*np.pi), np.arcsin(ground_Z)

