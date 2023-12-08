# this is the junk drawer of functions
import warnings
from datetime import datetime

import numpy as np
import pytz

from . import beam  # noqa F401
from . import constants  # noqa F401
from . import coords  # noqa F401
from . import functions  # noqa F401
from . import io  # noqa F401
from . import linalg  # noqa F401
from . import pointing  # noqa F401
from . import units  # noqa F401


def get_utc_day_hour(t):
    dt = datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc)
    return dt.hour + dt.minute / 60 + dt.second / 3600


def get_utc_year_day(t):
    tt = datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc).timetuple()
    return tt.tm_yday + get_utc_day_hour(t) / 24 - 1


def get_utc_year(t):
    return datetime.fromtimestamp(t, tz=pytz.utc).replace(tzinfo=pytz.utc).year


class PointingError(Exception):
    pass


def validate_pointing(azim, elev):
    el_min = np.atleast_1d(elev).min()
    if el_min < np.radians(10):
        warnings.warn(
            f"Some detectors come within 10 degrees of the horizon (el_min = {np.degrees(el_min):.01f}°)"
        )
    if el_min <= 0:
        raise PointingError(
            f"Some detectors are pointing below the horizon (el_min = {np.degrees(el_min):.01f}°)"
        )


# def get_pointing_offset(time, period, throws, plan_type):
#     if plan_type == "daisy":
#         phase = 2 * np.pi * time / period

#         k = np.pi  # this added an irrational precession to the daisy
#         r = np.sin(k * phase)

#         return throws[0] * r * np.cos(phase), throws[1] * r * np.sin(phase)


def get_daisy_offsets(phase, k=2.11):
    r = np.sin(k * phase)
    return r * np.cos(phase), r * np.sin(phase)


def daisy_pattern_miss_center(
    phase, petals=3, radius=1, offset_factor=0.15, k1=1 / 3 * np.pi, k2=2 / 3 * np.pi
):
    shifted_phase = phase + np.pi / 2
    z1 = np.sin(phase * k2) * np.exp(1j * phase / petals)
    z2 = (
        offset_factor * np.sin(shifted_phase * k1) * np.exp(1j * shifted_phase / petals)
    )
    z = z1 - z2

    max_dist = np.atleast_1d(np.abs(z)).max()
    z *= radius / max_dist


#         daisy_offset_x, daisy_offset_y = get_pointing_offset(time, period, plan_type="daisy")

#         return (
#             centers[0] + throws[0] * r * np.cos(p),
#             centers[1] + throws[1] * r * np.sin(p),
#         )

#     if plan_type == "box":
#         return (
#             centers[0]
#             + throws[0] * np.interp(p % (2 * np.pi), np.linspace(0, 2 * np.pi, 5), [-1, -1, +1, +1, -1]),
#             centers[1] + throws[1] * np.interp(p, np.linspace(0, 2 * np.pi, 5), [-1, +1, +1, -1, -1]),
#         )


# def get_daisy_offsets(phase, k=2.11):
#     r = np.sin(k * phase)
#     return r * np.cos(phase), r * np.sin(phase)


# def get_pointing(time, **kwargs):
#     """
#     Returns azimuth and elevation
#     """
#     scan_center = kwargs.get("scan_center", (np.radians(10), np.radians(4.5)))
#     scan_pattern = kwargs.get("scan_pattern", "stare")
#     scan_speed = kwargs.get("scan_speed", 60)
#     scan_radius = kwargs.get("scan_radius", np.radians(2))

#     phase = 2 * np.pi * time / scan_speed

#     if scan_pattern == "daisy":
#         dpox, dpoy = get_daisy_offsets(phase)
#         return xy_to_lonlat(   # noqa F401
#             scan_radius * dpox, scan_radius * dpoy, *scan_center
#         )


# # COORDINATE TRANSFORM UTILS
# class Planner:
#     def __init__(self, Array):
#         self.array = Array

#     def make_plans(self, start, end, ra, dec, chunk_time, static_config):
#         start_time = datetime.fromisoformat(start).replace(tzinfo=pytz.utc).timestamp()
#         end_time = datetime.fromisoformat(end).replace(tzinfo=pytz.utc).timestamp()

#         _unix = np.arange(start_time, end_time, chunk_time)
#         _ra = np.radians(np.linspace(ra, ra, len(_unix)))
#         _dec = np.radians(np.linspace(dec, dec, len(_unix)))

#         _az, _el = self.array.coordinator.transform(
#             _unix, _ra, _dec, in_frame="ra_dec", out_frame="az_el"
#         )

#         min_el = np.degrees(np.minimum(_el[1:], _el[:-1]))

#         ok = (min_el > self.array.el_bounds[0]) & (min_el < self.array.el_bounds[1])

#         self.time, self.az, self.el = _unix[1:][ok], _az[1:][ok], _el[1:][ok]

#         for start_time in _unix[:-1][ok]:
#             yield dict(
#                 {
#                     "start_time": start_time,
#                     "end_time": start_time + chunk_time,
#                     "coord_center": (ra, dec),
#                     "coord_throw": (2, 2),
#                     "coord_frame": "ra_dec",
#                 },
#                 **static_config,
#             )


# ================ ARRAY ================

# ================ STATS ================


# def get_minimal_bounding_rotation_angle(z):  # minimal-area rotation angle

#     H = sp.spatial.ConvexHull(points=np.vstack([np.real(z).ravel(), np.imag(z).ravel()]).T)
#     HZ = z.ravel()[H.vertices]

#     HE = np.imag(HZ).max() - np.imag(HZ).min()
#     HO = 0
#     # for z1,z2 in zip(HZ,np.roll(HZ,1)):
#     for RHO in np.linspace(0, np.pi, 1024 + 1)[1:]:

#         # RHO = np.angle(z2-z1)
#         RHZ = HZ * np.exp(1j * RHO)

#         im_width = np.imag(RHZ).max() - np.imag(RHZ).min()
#         re_width = np.real(RHZ).max() - np.real(RHZ).min()

#         RHE = im_width  # * re_width

#         if RHE < HE and re_width > im_width:
#             HE = RHE
#             HO = RHO

#     return HO


# def smallest_max_error_sample(items, max_error=1e0):

#     k = 1
#     cluster_mids = np.sort(sp.cluster.vq.kmeans(items, k_or_guess=1)[0])
#     while (np.abs(np.subtract.outer(items, cluster_mids)) / cluster_mids[None, :]).min(axis=1).max() > max_error:
#         cluster_mids = np.sort(sp.cluster.vq.kmeans(items, k_or_guess=k)[0])
#         k += 1

#     which_cluster = np.abs(np.subtract.outer(items, cluster_mids)).argmin(axis=1)

#     return cluster_mids, which_cluster


# def make_beam_filter(side, window_func, args):

#     beam_X, beam_Y = np.meshgrid(side, side)
#     beam_R = np.sqrt(np.square(beam_X) + np.square(beam_Y))
#     beam_W = window_func(beam_R, *args)

#     return beam_W / beam_W.sum()


# def get_brightness_temperature(f_pb, pb, f_spec, spec):

#     return sp.integrate.trapz(
#         sp.interpolate.interp1d(f_spec, spec, axis=-1)(f_pb) * pb, f_pb, axis=-1
#     ) / sp.integrate.trapz(pb, f_pb)


# ================ POINTING ================
