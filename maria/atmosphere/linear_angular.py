from typing import Sequence

import numpy as np
import scipy as sp
from tqdm import tqdm

from .. import utils
from .base import BaseAtmosphericSimulation, extrude
from .single_layer import SingleLayerSimulation

MIN_SAMPLES_PER_RIBBON = 2
RIBBON_SAMPLE_DECAY = 4
JITTER_LEVEL = 1e-4

matern_callback = utils.functions.approximate_normalized_matern
# matern_callback = utils.functions.normalized_matern


class LinearAngularSimulation(BaseAtmosphericSimulation):
    """
    The simplest possible atmospheric model.
    This model is only appropriate for single instruments, i.e. when the baseline is zero.
    """

    def __init__(
        self,
        array,
        pointing,
        site,
        min_layer_height=0,
        max_layer_height=3000,
        n_layers=4,
        min_atmosphere_beam_res=4,
        verbose=False,
        **kwargs,
    ):
        super().__init__(array, pointing, site, verbose=verbose, **kwargs)

        layer_boundaries = np.linspace(min_layer_height, max_layer_height, n_layers + 1)
        self.layer_heights = 0.5 * (layer_boundaries[1:] + layer_boundaries[:-1])
        self.n_layers = n_layers

        self.layer_sims = []
        for layer_height in self.layer_heights:
            self.layer_sims.append(
                SingleLayerSimulation(
                    array,
                    pointing,
                    site,
                    layer_height=layer_height,
                    min_atmosphere_beam_res=min_atmosphere_beam_res,
                    verbose=verbose,
                )
            )

    def simulate_integrated_water_vapor(self):
        layer_values = np.zeros(
            (self.array.n_dets, self.pointing.n_time, self.n_layers)
        )

        for i, sim in enumerate(self.layer_sims):
            layer_values[..., i] = sim.simulate_normalized_effective_water_vapor()

        rel_layer_scaling = np.interp(
            self.site.altitude + self.layer_heights,
            self.weather.altitude_levels,
            self.weather.absolute_humidity,
        )
        rel_layer_scaling /= np.sqrt(np.square(rel_layer_scaling).sum())

        self.layer_scaling = (
            self.site.pwv_rms_frac * self.weather.pwv * rel_layer_scaling
        )

        self.line_of_sight_pwv = (
            self.weather.pwv + layer_values @ self.layer_scaling
        ) / np.sin(self.EL)

        # n_steps = self.n_extrusion

        # extruded_values = extrude(values=self.shaped_values.ravel(),
        #                           A=self.A,
        #                           B=self.B,
        #                           n_steps=n_steps,
        #                           n_i=self.n_extrusion,
        #                           n_j=self.n_cross_section,
        #                           i_sample_index=self.extrusion_sample_index,
        #                           j_sample_index=self.cross_section_sample_index)

        # self.shaped_values = extruded_values.reshape(self.n_extrusion, self.n_cross_section)

        # trans_detector_angular_positions = (
        #     self.detector_offsets[:, None] + self.boresight_angular_position[None]
        # ) @ self.R.T
        # detector_values = np.zeros(trans_detector_angular_positions.shape[:-1])
        # for uband in self.array.ubands:
        #     band_mask = self.array.dets.band == uband
        #     band_angular_fwhm = self.angular_beam_fwhm[band_mask].mean()
        #     F = utils.beam.make_beam_filter(
        #         band_angular_fwhm, self.angular_resolution, self.array.beam_profile
        #     )
        #     FILTERED_VALUES = utils.beam.separably_filter(self.shaped_values, F)
        #     detector_values[band_mask] = sp.interpolate.RegularGridInterpolator(
        #         (self.cross_section_side, self.extrusion_side), FILTERED_VALUES.T
        #     )(trans_detector_angular_positions[band_mask])

        # # this is "zenith-scaled"
        # self.line_of_sight_pwv = (
        #     self.weather.pwv
        #     * (1.0 + self.site.pwv_rms_frac * detector_values)
        #     / np.sin(self.EL)
        # )


# class LinearAngularSimulation(BaseAtmosphericSimulation):
#     """
#     The linear angular model treats the atmosphere as a bunch of layers.

#     This model is only appropriate for single instruments, i.e. when the baseline is zero.
#     """

#     def __init__(
#         self, array, pointing, site, config=DEFAULT_LA_CONFIG, verbose=False, **kwargs
#     ):
#         super().__init__(array, pointing, site, **kwargs)

#         self.config = config
#         for key, val in config.items():
#             setattr(self, key, val)
#             if verbose:
#                 print(f"set {key} to {val}")

#         self.layer_boundaries = np.linspace(
#             self.min_depth, self.max_depth, self.n_layers + 1
#         )

#         # layer of integrated atmosphere
#         self.layer_depths = 0.5 * (
#             self.layer_boundaries[1:] + self.layer_boundaries[:-1]
#         )
#         self.layer_thicks = np.diff(self.layer_boundaries)

#         # returns a beam waists and angular waists for each frequency for each layer depth
#         self.waists = self.array.get_beam_waist(
#             self.layer_depths[:, None],
#             self.array.primary_size,
#             self.array.dets.band_center.values,
#         )
#         self.angular_waists = self.waists / self.layer_depths[:, None]

#         self.min_ang_res = self.angular_waists / self.min_beam_res

#         self.weather.generate(
#             time=self.pointing.time, fixed_quantiles=self.site.weather_quantiles
#         )

#         self.heights = (
#             self.site.altitude
#             + self.layer_depths[:, None] * np.sin(self.pointing.el)[None, :]
#         )

#         for attr in [
#             "water_vapor",
#             "temperature",
#             "pressure",
#             "wind_north",
#             "wind_east",
#         ]:
#             setattr(
#                 self,
#                 attr,
#                 sp.interpolate.RegularGridInterpolator(
#                     (self.weather.levels, self.weather.time),
#                     getattr(self.weather, attr),
#                 )((self.heights, self.pointing.time[None])),
#             )

#             # setattr(self, attr, sp.interpolate.interp1d(self.weather.height, getattr(self.weather, attr), axis=0)(self.heights))

#         self.wind_bearing = np.arctan2(self.wind_east, self.wind_north)
#         self.wind_speed = np.sqrt(
#             np.square(self.wind_east) + np.square(self.wind_north)
#         )

#         self.layer_scaling = (
#             self.water_vapor
#             * self.temperature
#             * self.wind_speed
#             * self.layer_thicks[:, None]
#         )
#         self.layer_scaling = np.sqrt(
#             np.square(self.layer_scaling) / np.square(self.layer_scaling).sum(axis=0)
#         )

#         # the velocity of the atmosphere with respect to the array pointing
#         self.AWV_X = (
#             +self.wind_east * np.cos(self.pointing.az[None])
#             - self.wind_north * np.sin(self.pointing.az[None])
#         ) / self.layer_depths[:, None]
#         self.AWV_Y = (
#             (
#                 -self.wind_east * np.sin(self.pointing.az[None])
#                 + self.wind_north * np.cos(self.pointing.az[None])
#             )
#             / self.layer_depths[:, None]
#             * np.sin(self.pointing.el[None])
#         )

#         center_lon, center_lat = utils.coords.get_center_lonlat(
#             self.pointing.az, self.pointing.el
#         )
#         self.pointing.dx, self.pointing.dy = utils.coords.lonlat_to_xy(
#             self.pointing.az, self.pointing.el, center_lon, center_lat
#         )

#         # the angular position of each detector over time WRT the atmosphere
#         self.REL_X = (
#             self.array.offset_x[None, :, None]
#             + self.pointing.dx[None, None]
#             + np.cumsum(self.AWV_X * self.pointing.dt, axis=-1)[:, None]
#         )
#         self.REL_Y = (
#             self.array.offset_y[None, :, None]
#             + self.pointing.dy[None, None]
#             + np.cumsum(self.AWV_Y * self.pointing.dt, axis=-1)[:, None]
#         )

#         self.MEAN_REL_X, self.MEAN_REL_Y = self.REL_X.mean(axis=1), self.REL_Y.mean(
#             axis=1
#         )

#         # These are empty lists we need to fill with chunky parameters (they won't fit together!) for each layer.
#         self.para, self.orth, self.P, self.O, self.X, self.Y = [], [], [], [], [], []
#         self.n_para, self.n_orth, self.lay_ang_res, self.genz, self.AR_samples = (
#             [],
#             [],
#             [],
#             [],
#             [],
#         )

#         self.p = np.zeros((self.REL_X.shape))
#         self.o = np.zeros((self.REL_X.shape))

#         self.layer_rotation_angles = []
#         self.outer_scale = 3e2
#         self.ang_outer_scale = self.outer_scale / self.layer_depths

#         self.theta_edge_z = []

#         radius_sample_prop = 1.5
#         beam_tol = 1e-2

#         max_layer_beam_radii = 0.5 * self.angular_waists.max(axis=1)

#         self.padding = (radius_sample_prop + beam_tol) * max_layer_beam_radii

#         for i_l, depth in enumerate(self.layer_depths):
#             MEAN_REL_POINTS = np.c_[self.MEAN_REL_X[i_l], self.MEAN_REL_Y[i_l]]
#             MEAN_REL_POINTS += 1e-12 * np.random.standard_normal(
#                 size=MEAN_REL_POINTS.shape
#             )  # some jitter for the hull

#             hull = sp.spatial.ConvexHull(MEAN_REL_POINTS)
#             h_x, h_y = hull.points[hull.vertices].T
#             h_z = h_x + 1j * h_y
#             layer_hull_theta_z = h_z * (np.abs(h_z) + self.padding[i_l]) / np.abs(h_z)

#             self.layer_rotation_angles.append(
#                 utils.get_minimal_bounding_rotation_angle(layer_hull_theta_z.ravel())
#             )

#             zop = (self.REL_X[i_l] + 1j * self.REL_Y[i_l]) * np.exp(
#                 1j * self.layer_rotation_angles[-1]
#             )
#             self.p[i_l], self.o[i_l] = np.real(zop), np.imag(zop)

#             res = self.min_ang_res[i_l].min()

#             para_ = np.arange(
#                 self.p[i_l].min() - self.padding[i_l] - res,
#                 self.p[i_l].max() + self.padding[i_l] + res,
#                 res,
#             )
#             orth_ = np.arange(
#                 self.o[i_l].min() - self.padding[i_l] - res,
#                 self.o[i_l].max() + self.padding[i_l] + res,
#                 res,
#             )

#             n_para, n_orth = len(para_), len(orth_)
#             self.lay_ang_res.append(res)

#             self.PARA_SPACING = np.gradient(para_).mean()
#             self.para.append(para_), self.orth.append(orth_)
#             self.n_para.append(len(para_)), self.n_orth.append(len(orth_))

#             ORTH_, PARA_ = np.meshgrid(orth_, para_)

#             self.genz.append(
#                 np.exp(-1j * self.layer_rotation_angles[-1])
#                 * (PARA_[0] + 1j * ORTH_[0] - res)
#             )
#             XYZ = np.exp(-1j * self.layer_rotation_angles[-1]) * (PARA_ + 1j * ORTH_)

#             self.X.append(np.real(XYZ)), self.Y.append(np.imag(XYZ))
#             # self.O.append(ORTH_), self.P.append(PARA_)

#             del zop

#             para_i, orth_i = [], []
#             for ii, i in enumerate(
#                 np.r_[
#                     0,
#                     2 ** np.arange(np.ceil(np.log(self.n_para[-1]) / np.log(2))),
#                     self.n_para[-1] - 1,
#                 ]
#             ):
#                 orth_i.append(
#                     np.unique(
#                         np.linspace(
#                             0,
#                             self.n_orth[-1] - 1,
#                             int(np.maximum(self.n_orth[-1] / (4**ii), 4)),
#                         ).astype(int)
#                     )
#                 )
#                 para_i.append(np.repeat(i, len(orth_i[-1])).astype(int))

#             self.AR_samples.append((np.concatenate(para_i), np.concatenate(orth_i)))

#             n_cm = len(self.AR_samples[-1][0])

#             if n_cm > 5000 and verbose:
#                 warning_message = f"A very large covariance matrix for layer {i_l+1} (n_side = {n_cm})"
#                 warnings.warn(warning_message)

#         self.C00, self.C01, self.C11, self.A, self.B = [], [], [], [], []

#         with tqdm(total=len(self.layer_depths), desc="Computing weights") as prog:
#             for i_l, (depth, LX, LY, AR, GZ) in enumerate(
#                 zip(self.layer_depths, self.X, self.Y, self.AR_samples, self.genz)
#             ):
#                 X0, Y0 = LX[AR], LY[AR]
#                 X1, Y1 = np.real(GZ), np.imag(GZ)

#                 R00 = np.sqrt(
#                     np.subtract.outer(X0, X0) ** 2 + np.subtract.outer(Y0, Y0) ** 2
#                 )
#                 R01 = np.sqrt(
#                     np.subtract.outer(X1, X0) ** 2 + np.subtract.outer(Y1, Y0) ** 2
#                 )
#                 R11 = np.sqrt(
#                     np.subtract.outer(X1, X1) ** 2 + np.subtract.outer(Y1, Y1) ** 2
#                 )

#                 alpha = 1e-4

#                 C00 = utils.approximate_matern(
#                     R00, self.outer_scale / depth, 5 / 6
#                 ) + alpha * np.eye(len(X0))
#                 C01 = utils.approximate_matern(R01, self.outer_scale / depth, 5 / 6)
#                 C11 = utils.approximate_matern(
#                     R11, self.outer_scale / depth, 5 / 6
#                 ) + alpha * np.eye(len(X1))

#                 self.C00.append(C00)
#                 self.C01.append(C01)
#                 self.C11.append(C11)
#                 self.A.append(np.matmul(C01, utils.linalg.fast_psd_inverse(C00)))
#                 self.B.append(
#                     sp.linalg.lapack.dpotrf(C11 - np.matmul(self.A[-1], C01.T))[0]
#                 )

#                 prog.update(1)

#             if verbose:
#                 print(
#                     "\n # | depth (m) | beam (m) | beam (') | sim (m) | sim (') | rms (mg/m2) | n_cov | orth | para | h2o (g/m3) | temp (K) | ws (m/s) | wb (deg) |"
#                 )

#                 for i_l, depth in enumerate(self.layer_depths):
#                     row_string = f"{i_l+1:2} | {depth:9.01f} | {self.waists[i_l].min():8.02f} | {60*np.degrees(self.angular_waists[i_l].min()):8.02f} | "
#                     row_string += f"{depth*self.lay_ang_res[i_l]:7.02f} | {60*np.degrees(self.lay_ang_res[i_l]):7.02f} | "
#                     row_string += f"{1e3*self.layer_scaling[i_l].mean():11.02f} | {len(self.AR_samples[i_l][0]):5} | {self.n_orth[i_l]:4} | "
#                     row_string += f"{self.n_para[i_l]:4} | {1e3*self.water_vapor[i_l].mean():11.02f} | {self.temperature[i_l].mean():8.02f} | "
#                     row_string += f"{self.wind_speed[i_l].mean():8.02f} | {np.degrees(self.wind_bearing[i_l].mean()+np.pi):8.02f} |"
#                     print(row_string)

#     def atmosphere_timestep(
#         self, i
#     ):  # iterate the i-th layer of atmosphere by one step
#         self.vals[i] = np.r_[
#             (
#                 np.matmul(self.A[i], self.vals[i][self.AR_samples[i]])
#                 + np.matmul(self.B[i], np.random.standard_normal(self.B[i].shape[0]))
#             )[None, :],
#             self.vals[i][:-1],
#         ]

#     def initialize_atmosphere(self, blurred=False):
#         self.vals = [np.zeros(lx.shape) for lx in self.X]
#         n_init_ = [n_para for n_para in self.n_para]
#         n_ts_ = [n_para for n_para in self.n_para]
#         tot_n_init, tot_n_ts = np.sum(n_init_), np.sum(n_ts_)
#         # self.gen_data = [np.zeros((n_ts,v.shape[1])) for n_ts,v in zip(n_ts_,self.lay_v_)]

#         with tqdm(total=tot_n_init, desc="Generating layers") as prog:
#             for i, n_init in enumerate(n_init_):
#                 for i_init in range(n_init):
#                     self.atmosphere_timestep(i)
#                     prog.update(1)

#     def simulate_integrated_water_vapor(self, do_atmosphere=True, verbose=False):
#         self.sim_start = ttime.time()
#         self.initialize_atmosphere()

#         self.rel_flucs = np.zeros(self.o.shape)

#         multichromatic_beams = False

#         with tqdm(total=self.n_layers, desc="Sampling layers") as prog:
#             for i_d, d in enumerate(self.layer_depths):
#                 # Compute the filtered line-of-sight pwv corresponding to each layer

#                 if multichromatic_beams:
#                     filtered_vals = np.zeros(
#                         (len(self.array.nu), *self.vals[i_d].shape)
#                     )
#                     angular_res = self.lay_ang_res[i_d]

#                     for i_f, f in enumerate(self.array.nu):
#                         angular_waist = self.angular_waists[i_d, i_f]

#                         self.F = self.array.make_filter(
#                             angular_waist, angular_res, self.array.beam_func
#                         )
#                         u, s, v = self.array.separate_filter(self.F)

#                         filtered_vals[i_f] = self.array.separably_filter(
#                             self.vals[i_d], u, s, v
#                         )

#                 else:
#                     angular_res = self.lay_ang_res[i_d]
#                     angular_waist = self.angular_waists[i_d].mean()

#                     self.F = self.array.make_filter(
#                         angular_waist, angular_res, self.array.beam_func
#                     )

#                     filtered_vals = self.array.separably_filter(self.vals[i_d], self.F)
#                     self.rel_flucs[i_d] = sp.interpolate.RegularGridInterpolator(
#                         (self.para[i_d], self.orth[i_d]), filtered_vals
#                     )((self.p[i_d], self.o[i_d]))

#                 prog.update(1)

#         # Convert PWV fluctuations to detector powers

#         self.effective_zenith_water_vapor = (
#             self.rel_flucs * self.layer_scaling[:, None]
#         ).sum(axis=0)

#         self.effective_zenith_water_vapor *= (
#             2e-2
#             * self.weather.column_water_vapor
#             / self.effective_zenith_water_vapor.std()
#         )
#         self.effective_zenith_water_vapor += self.weather.column_water_vapor.mean()

#         self.integrated_water_vapor = self.effective_zenith_water_vapor / np.sin(
#             self.EL
#         )

#         # self.atm_power = np.zeros(self.effective_zenith_water_vapor.shape)

#         # with tqdm(total=len(self.array.ubands), desc='Integrating spectra') as prog:
#         #     for b in self.array.ubands:

#         #         bm = self.array.dets.bands == b

#         #         ba_am_trj = (self.am.t_rj * self.array.am_passbands[bm].mean(axis=0)[None,None,:]).sum(axis=-1)

#         #         BA_TRJ_RGI = sp.interpolate.RegularGridInterpolator((self.am.tcwv, np.radians(self.am.elev)), ba_am_trj)

#         #         self.atm_power[bm] = BA_TRJ_RGI((self.epwv[bm], self.EL[bm]))

#         #         prog.update(1)


# def get_extrusion_products(
#     time,
#     azim,
#     elev,
#     baseline,
#     field_of_view,
#     min_frequency,
#     wind_velocity,
#     wind_direction,
#     max_depth=3000,
#     extrusion_step=1,
#     min_res=10,
#     max_res=100,
# ):
#     """
#     For the Kolomogorov-Taylor model. It works the same as one of those pasta extruding machines (https://youtu.be/TXtm_eNaIwQ). This function figures out the shape of the hole.

#     The inputs are:

#     azim, elev: time-ordered pointing
#     """

#     # Within this function, we work in the frame of wind-relative pointing phi'. The elevation stays the same. We define:
#     #
#     # phi' -> phi - phi_w
#     #
#     # where phi_w is the wind bearing, i.e. the direction the wind comes from. This means that:
#     #
#     # For phi' = 0°  you are looking into the wind
#     # For phi' = 90° the wind is going from left to right in your field of view.
#     #
#     # We further define the frame (x, y, z) = (r cos phi' sin theta, r sin phi' sin theta, r sin theta)

#     n_baseline = len(baseline)

#     # this converts the $(x, y, z)$ vertices of a beam looking straight up to the $(x, y, z)$ vertices of
#     # the time-ordered pointing of the beam in the frame of the wind direction.
#     elev_rotation_matrix = sp.spatial.transform.Rotation.from_euler(
#         "y", np.pi / 2 - elev
#     ).as_matrix()
#     azim_rotation_matrix = sp.spatial.transform.Rotation.from_euler(
#         "z", np.pi / 2 - azim + wind_direction
#     ).as_matrix()
#     total_rotation_matrix = np.matmul(azim_rotation_matrix, elev_rotation_matrix)

#     # this defines the maximum size of the beam that we have to worry about
#     max_beam_radius = _beam_sigma(max_depth, primary_size=primary_sizes, nu=5e9)
#     max_boresight_offset = max_beam_radius + max_depth * field_of_view / 2

#     # this returns a list of time-ordered vertices, which can used to construct a convex hull
#     time_ordered_vertices_list = []
#     for _min_radius, _max_radius, _baseline in zip(
#         primary_sizes / 2, max_boresight_offset, baseline
#     ):
#         _rot_baseline = np.matmul(
#             sp.spatial.transform.Rotation.from_euler("z", wind_direction).as_matrix(),
#             _baseline,
#         )

#         _vertices = np.c_[
#             np.r_[
#                 [
#                     _.ravel()
#                     for _ in np.meshgrid(
#                         [-_min_radius, _min_radius], [-_min_radius, _min_radius], [0]
#                     )
#                 ]
#             ],
#             np.r_[
#                 [
#                     _.ravel()
#                     for _ in np.meshgrid(
#                         [-_max_radius, _max_radius],
#                         [-_max_radius, _max_radius],
#                         [max_depth],
#                     )
#                 ]
#             ],
#         ]

#         time_ordered_vertices_list.append(
#             _rot_baseline[None]
#             + np.swapaxes(np.matmul(total_rotation_matrix, _vertices), 1, -1).reshape(
#                 -1, 3
#             )
#         )

#     # these are the bounds in space that we have to worry about. it has shape (3, 2)
#     extrusion_bounds = np.c_[
#         np.min([np.min(tov, axis=0) for tov in time_ordered_vertices_list], axis=0),
#         np.max([np.max(tov, axis=0) for tov in time_ordered_vertices_list], axis=0),
#     ]

#     # here we make the layers of the atmosphere given the bounds of $z$ and the specified resolution; we use
#     # regular layers to make interpolation more efficient later
#     min_height = np.max(
#         [
#             np.max(tov[:, 2][tov[:, 2] < np.median(tov[:, 2])])
#             for tov in time_ordered_vertices_list
#         ]
#     )
#     max_height = np.min(
#         [
#             np.min(tov[:, 2][tov[:, 2] > np.median(tov[:, 2])])
#             for tov in time_ordered_vertices_list
#         ]
#     )

#     height_samples = np.linspace(min_height, max_height, 1024)
#     dh = np.gradient(height_samples).mean()
#     dheight_dindex = np.interp(
#         height_samples, [min_height, max_height], [min_res, max_res]
#     )
#     dindex_dheight = 1 / dheight_dindex
#     n_layers = int(np.sum(dindex_dheight * dh))
#     layer_heights = sp.interpolate.interp1d(
#         np.cumsum(dindex_dheight * dh),
#         height_samples,
#         bounds_error=False,
#         fill_value="extrapolate",
#     )(1 + np.arange(n_layers))

#     # here we define the cells through which turbulence will be extruded.
#     layer_res = np.gradient(layer_heights)
#     x_min, x_max = extrusion_bounds[0, 0], extrusion_bounds[0, 1]
#     n_per_layer = ((x_max - x_min) / layer_res).astype(int)
#     cell_x = np.concatenate(
#         [
#             np.linspace(x_min, x_max, n)
#             for i, (res, n) in enumerate(zip(layer_res, n_per_layer))
#         ]
#     )
#     cell_z = np.concatenate(
#         [h * np.ones(n) for i, (h, n) in enumerate(zip(layer_heights, n_per_layer))]
#     )
#     cell_res = np.concatenate(
#         [res * np.ones(n) for i, (res, n) in enumerate(zip(layer_res, n_per_layer))]
#     )

#     extrusion_cells = np.c_[cell_x, cell_z]
#     extrusion_shift = np.c_[cell_res, np.zeros(len(cell_z))]

#     eps = 1e-6

#     in_view = np.zeros(len(extrusion_cells)).astype(bool)
#     for tov in time_ordered_vertices_list:
#         baseline_in_view = np.zeros(len(extrusion_cells)).astype(bool)

#         hull = sp.spatial.ConvexHull(
#             tov[:, [0, 2]]
#         )  # we only care about the $(x, z)$ dimensions here
#         A, b = hull.equations[:, :-1], hull.equations[:, -1:]

#         for shift_factor in [-1, 0, +1]:
#             baseline_in_view |= np.all(
#                 (extrusion_cells + shift_factor * extrusion_shift) @ A.T + b.T < eps,
#                 axis=1,
#             )

#         in_view |= (
#             baseline_in_view  # if the cell is in view of any of the baselines, keep it!
#         )

#         # plt.scatter(tov[:, 0], tov[:, 2], s=1e0)

#     # here we define the extrusion axis
#     y_min, y_max = (
#         extrusion_bounds[1, 0],
#         extrusion_bounds[1, 1] + wind_velocity * time.ptp(),
#     )
#     extrustion_axis = np.arange(y_min, y_max, extrusion_step)

#     return extrusion_cells[in_view].T, extrustion_axis, cell_res[in_view]
