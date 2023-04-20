# -*- coding: utf-8 -*-
import warnings
import pytz
import numpy as np
import scipy as sp
import astropy as ap
import astropy.constants as const
from astropy import units as u

from numpy import linalg as la
from datetime import datetime



class PointingError(Exception):
    pass

def validate_pointing(azim, elev):
    el_min = np.atleast_1d(elev).min()
    if el_min < np.radians(10):
        warnings.warn(f"Some detectors come within 10 degrees of the horizon (el_min = {np.degrees(el_min):.01f}°)")
    if el_min <= 0:
        raise PointingError(f"Some detectors are pointing below the horizon (el_min = {np.degrees(el_min):.01f}°)")



def _beam_sigma(z, primary_size, nu):

    c = 2.998e8
    n = 1
    w0 = primary_size / np.sqrt(2 * np.log(2))
    zR = np.pi * w0**2 * n * nu / c

    return 0.5 * w0 * np.sqrt(1 + (z / zR) ** 2)


def normalized_matern(r, r0, nu):

    nu = 5 / 6
    d_eff = np.abs(r) / r0

    return (
        2 ** (1 - nu)
        * sp.special.kv(nu, np.sqrt(2 * nu) * d_eff + 1e-6)
        * (np.sqrt(2 * nu) * d_eff + 1e-6) ** nu
        / sp.special.gamma(nu)
    )


def _approximate_normalized_matern(r, r0, nu, n_test_points=4096):
    """
    Computing BesselK[nu,z] for arbitrary nu is expensive. This is good for casting over huge matrices.
    """

    _r = np.atleast_1d(np.abs(r))

    r_min = np.minimum(1e-6 * r0, _r[_r > 0].min())
    r_max = np.maximum(1e02 * r0, _r.max())

    rrel = np.cumsum(np.linspace(0, 1, n_test_points) ** 4)
    r_test = rrel * (r_max - r_min) / rrel.max() + r_min

    return np.exp(np.interp(np.abs(r), r_test, np.log(normalized_matern(r_test, r0, 1 / 3))))


def _fast_psd_inverse(M):
    """
    About twice as fast as np.linalg.inv for large, PSD matrices.
    """

    cholesky, dpotrf_info = sp.linalg.lapack.dpotrf(M)
    invM, dpotri_info = sp.linalg.lapack.dpotri(cholesky)
    return np.where(invM, invM, invM.T)


def get_pointing(time, period, centers, throws, plan_type):

    p = 2 * np.pi * time / period

    if plan_type == "back-and-forth":
        return (
            centers[0] + throws[0] * sp.signal.sawtooth(p, width=0.5),
            centers[1] + throws[1] * sp.signal.sawtooth(p, width=0.5),
        )

    if plan_type == "daisy":

        k = np.pi  # this added an irrational precession to the daisy
        r = np.sin(k * p)

        return (
            centers[0] + throws[0] * r * np.cos(p),
            centers[1] + throws[1] * r * np.sin(p),
        )

    if plan_type == "box":
        return (
            centers[0]
            + throws[0] * np.interp(p % (2 * np.pi), np.linspace(0, 2 * np.pi, 5), [-1, -1, +1, +1, -1]),
            centers[1] + throws[1] * np.interp(p, np.linspace(0, 2 * np.pi, 5), [-1, +1, +1, -1, -1]),
        )


def datetime_handler(time):

    """
    Accepts any time format you can think of, spits out datetime object
    """
    if isinstance(time, (int, float)):
        return datetime.fromtimestamp(time).astimezone(pytz.utc)
    if isinstance(time, str):
        return datetime.fromisoformat(time).replace(tzinfo=pytz.utc)


# COORDINATE TRANSFORM UTILS
class Planner:
    def __init__(self, Array):

        self.array = Array

    def make_plans(self, start, end, ra, dec, chunk_time, static_config):

        start_time = datetime.fromisoformat(start).replace(tzinfo=pytz.utc).timestamp()
        end_time = datetime.fromisoformat(end).replace(tzinfo=pytz.utc).timestamp()

        _unix = np.arange(start_time, end_time, chunk_time)
        _ra = np.radians(np.linspace(ra, ra, len(_unix)))
        _dec = np.radians(np.linspace(dec, dec, len(_unix)))

        _az, _el = self.array.coordinator.transform(_unix, _ra, _dec, in_frame="ra_dec", out_frame="az_el")

        min_el = np.degrees(np.minimum(_el[1:], _el[:-1]))

        ok = (min_el > self.array.el_bounds[0]) & (min_el < self.array.el_bounds[1])

        self.unix, self.az, self.el = _unix[1:][ok], _az[1:][ok], _el[1:][ok]

        for start_time in _unix[:-1][ok]:

            yield dict(
                {
                    "start_time": start_time,
                    "end_time": start_time + chunk_time,
                    "coord_center": (ra, dec),
                    "coord_throw": (2, 2),
                    "coord_frame": "ra_dec",
                },
                **static_config,
            )


def validate_args(DICT, necessary_args):
    missing_args = []
    for arg in necessary_args:
        if not arg in DICT:
            missing_args.append(arg)
    if not len(missing_args) == 0:
        raise Exception(f"missing arguments {missing_args}")


# ================ ARRAY ================


def get_passband(nu, nu_0, nu_w, order=4):
    return np.exp(-np.abs((nu - nu_0) / (nu_w / 2)) ** order)


def make_array(array_shape, max_fov, n_det):

    valid_array_types = ["flower", "hex", "square"]

    if array_shape == "flower":
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        dzs = np.zeros(n_det).astype(complex)
        for i in range(n_det):
            dzs[i] = np.sqrt((i / (n_det - 1)) * 2) * np.exp(1j * phi * i)
        od = np.abs(np.subtract.outer(dzs, dzs))
        dzs *= max_fov / od.max()
        return np.c_[np.real(dzs), np.imag(dzs)]
    if array_shape == "hex":
        return make_hex(n_det, max_fov)
    if array_shape == "square":
        dxy_ = np.linspace(-max_fov, max_fov, int(np.ceil(np.sqrt(n_det)))) / (2 * np.sqrt(2))
        DX, DY = np.meshgrid(dxy_, dxy_)
        return np.c_[DX.ravel()[:n_det], DY.ravel()[:n_det]]

    raise ValueError("Please specify a valid array type. Valid array types are:\n" + "\n".join(valid_array_types))


def make_hex(n, d):

    angles = np.linspace(0, 2 * np.pi, 6 + 1)[1:] + np.pi / 2
    zs = np.array([0])
    layer = 0
    while len(zs) < n:
        for angle in angles:
            for z in layer * np.exp(1j * angle) + np.arange(layer) * np.exp(1j * (angle + 2 * np.pi / 3)):
                zs = np.append(zs, z)
        layer += 1
    zs -= zs.mean()
    zs *= 0.5 * d / np.abs(zs).max()

    return np.c_[np.real(np.array(zs[:n])), np.imag(np.array(zs[:n]))]


# ================ STATS ================

msqrt = lambda M: [np.matmul(u, np.diag(np.sqrt(s))) for u, s, vh in [la.svd(M)]][0]
matern = (
    lambda r, r0, nu: 2 ** (1 - nu)
    / sp.special.gamma(nu)
    * sp.special.kv(nu, r / r0 + 1e-10)
    * (r / r0 + 1e-10) ** nu
)

gaussian_beam = lambda z, w0, l, n: np.sqrt(1 / np.square(z) + np.square(l) / np.square(w0 * np.pi * n))


def get_minimal_bounding_rotation_angle(z):  # minimal-area rotation angle

    H = sp.spatial.ConvexHull(points=np.vstack([np.real(z).ravel(), np.imag(z).ravel()]).T)
    HZ = z.ravel()[H.vertices]

    HE = np.imag(HZ).max() - np.imag(HZ).min()
    HO = 0
    # for z1,z2 in zip(HZ,np.roll(HZ,1)):
    for RHO in np.linspace(0, np.pi, 1024 + 1)[1:]:

        # RHO = np.angle(z2-z1)
        RHZ = HZ * np.exp(1j * RHO)

        im_width = np.imag(RHZ).max() - np.imag(RHZ).min()
        re_width = np.real(RHZ).max() - np.real(RHZ).min()

        RHE = im_width  # * re_width

        if RHE < HE and re_width > im_width:
            HE = RHE
            HO = RHO

    return HO


def smallest_max_error_sample(items, max_error=1e0):

    k = 1
    cluster_mids = np.sort(sp.cluster.vq.kmeans(items, k_or_guess=1)[0])
    while (np.abs(np.subtract.outer(items, cluster_mids)) / cluster_mids[None, :]).min(axis=1).max() > max_error:
        cluster_mids = np.sort(sp.cluster.vq.kmeans(items, k_or_guess=k)[0])
        k += 1

    which_cluster = np.abs(np.subtract.outer(items, cluster_mids)).argmin(axis=1)

    return cluster_mids, which_cluster


def make_beam_filter(side, window_func, args):

    beam_X, beam_Y = np.meshgrid(side, side)
    beam_R = np.sqrt(np.square(beam_X) + np.square(beam_Y))
    beam_W = window_func(beam_R, *args)

    return beam_W / beam_W.sum()


def make_2d_covariance_matrix(C, args, x0, y0, x1=None, y1=None, auto=True):
    if auto:
        n = len(x0)
        i, j = np.triu_indices(n, 1)
        o = C(np.sqrt(np.square(x0[i] - x0[j]) + np.square(y0[i] - y0[j])), *args)
        c = np.empty((n, n))
        c[i, j], c[j, i] = o, o
        c[np.eye(n).astype(bool)] = 1
    if not auto:
        n = len(x0)
        i, j = np.triu_indices(n, 1)
        c = C(
            np.sqrt(np.square(np.subtract.outer(x0, x1)) + np.square(np.subtract.outer(y0, y1))),
            *args,
        )
    return c


def get_sub_splits(time_, xvel_, durations=[]):

    # flag wherever the scan velocity changing direction (can be more general)
    flags = np.r_[0, np.where(np.sign(xvel_[:-1]) != np.sign(xvel_[1:]))[0], len(xvel_) - 1]
    splits = np.array([[s, e] for s, e in zip(flags, flags[1:])]).astype(int)

    # compiles sub-scans that cover the TOD
    sub_splits = splits.copy()
    dt = np.median(np.gradient(time_))
    for i, (s, e) in enumerate(splits):

        split_dur = time_[e] - time_[s]
        for pld in durations:
            if pld > split_dur:
                continue
            sub_n = 2 * int(split_dur / pld) + 1
            sub_s = np.linspace(s, e - int(pld / dt), sub_n).astype(int)
            for sub_s in np.linspace(s, e - int(np.ceil(pld / dt)), sub_n).astype(int):
                sub_splits = np.r_[sub_splits, np.array([sub_s, sub_s + int(pld / dt)])[None, :]]

    return sub_splits


def get_brightness_temperature(f_pb, pb, f_spec, spec):

    return sp.integrate.trapz(
        sp.interpolate.interp1d(f_spec, spec, axis=-1)(f_pb) * pb, f_pb, axis=-1
    ) / sp.integrate.trapz(pb, f_pb)


# ================ POINTING ================


def to_xy(p, t, c_p, c_t):
    ground_X, ground_Y, ground_Z = (
        np.sin(p - c_p) * np.cos(t),
        np.cos(p - c_p) * np.cos(t),
        np.sin(t),
    )
    return np.arcsin(ground_X), np.arcsin(-np.real((ground_Y + 1j * ground_Z) * np.exp(1j * (np.pi / 2 - c_t))))


def from_xy(dx, dy, c_p, c_t):
    ground_X, Y, Z = (
        np.sin(dx + 1e-16),
        -np.sin(dy + 1e-16),
        np.cos(np.sqrt(dx**2 + dy**2)),
    )
    gyz = (Y + 1j * Z) * np.exp(-1j * (np.pi / 2 - c_t))
    ground_Y, ground_Z = np.real(gyz), np.imag(gyz)
    return (np.angle(ground_Y + 1j * ground_X) + c_p) % (2 * np.pi), np.arcsin(ground_Z)


# Kelvin CMB to Jy/pixel
# ----------------------------------------------------------------------
global Tcmb
Tcmb = 2.7255


def getJynorm():
    factor = 2e26
    factor *= (const.k_B * Tcmb * u.Kelvin) ** 3  # (kboltz*Tcmb)**3.0
    factor /= (const.h * const.c) ** 2  # (hplanck*clight)**2.0
    return factor.value


def getx(freq):
    factor = const.h * freq * u.Hz / const.k_B / (Tcmb * u.Kelvin)
    return factor.to(u.dimensionless_unscaled).value


def KcmbToJyPix(freq, ipix, jpix):
    x = getx(freq)
    factor = getJynorm() / Tcmb
    factor *= (x**4) * np.exp(x) / (np.expm1(x) ** 2)
    factor *= np.abs(ipix * jpix) * (np.pi / 1.8e2) * (np.pi / 1.8e2)
    return factor


def KcmbToJy(freq):
    x = getx(freq)
    factor = getJynorm() / Tcmb
    factor *= (x**4) * np.exp(x) / (np.expm1(x) ** 2)
    return factor

# Kelvin CMB to Kelvin brightness
# ----------------------------------------------------------------------
def KcmbToKbright(freq): 
  x = getx(freq)
  return np.exp(x)*((x/np.expm1(x))**2)

# Kelvin brightness to Jy/pixel
# ----------------------------------------------------------------------
def KbrightToJyPix(freq,ipix,jpix):
  return KcmbToJyPix(freq,ipix,jpix)/KcmbToKbright(freq)

# Kelvin CMB to Jy/pixel
# ----------------------------------------------------------------------
def KcmbToJyPix(freq,ipix,jpix):
  x = getx(freq)
  factor  = getJynorm()/Tcmb
  factor *= (x**4)*np.exp(x)/(np.expm1(x)**2)
  factor *= np.abs(ipix*jpix)*(np.pi/1.8e2)*(np.pi/1.8e2)
  return factor