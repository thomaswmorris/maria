import numpy as np
import scipy as sp

from .. import utils
from .base import BaseAtmosphericSimulation, extrude

MIN_SAMPLES_PER_RIBBON = 2
RIBBON_SAMPLE_DECAY = 4
JITTER_LEVEL = 1e-4

matern_callback = utils.functions.approximate_normalized_matern
# matern_callback = utils.functions.normalized_matern


class SingleLayerSimulation(BaseAtmosphericSimulation):
    """
    The simplest possible atmospheric model.
    This model is only appropriate for single instruments, i.e. when the baseline is zero.
    """

    def __init__(
        self,
        array,
        pointing,
        site,
        layer_height=1e3,
        min_atmosphere_beam_res=4,
        verbose=False,
        **kwargs,
    ):
        super().__init__(array, pointing, site, verbose=verbose, **kwargs)

        self.min_beam_res = min_atmosphere_beam_res
        self.layer_height = layer_height

        # this is approximately correct
        self.layer_depth = self.layer_height / np.sin(np.mean(self.pointing.el))

        # this might change
        self.angular_outer_scale = 500 / self.layer_depth

        # returns the beam fwhm for each detector at the layer distance
        self.physical_beam_fwhm = self.array.physical_fwhm(self.layer_height)
        self.angular_beam_fwhm = self.physical_beam_fwhm / self.layer_depth

        self.angular_resolution = self.angular_beam_fwhm.min() / self.min_beam_res
        if verbose:
            print(f"{self.angular_resolution = }")

        layer_altitude = self.site.altitude + self.layer_height

        layer_wind_north = sp.interpolate.interp1d(
            self.weather.altitude_levels, self.weather.wind_north, axis=0
        )(layer_altitude)
        layer_wind_east = sp.interpolate.interp1d(
            self.weather.altitude_levels, self.weather.wind_east, axis=0
        )(layer_altitude)

        angular_velocity_x = (
            +layer_wind_east * np.cos(self.pointing.az)
            - layer_wind_north * np.sin(self.pointing.az)
        ) / self.layer_depth

        angular_velocity_y = (
            -layer_wind_east * np.sin(self.pointing.az)
            + layer_wind_north * np.cos(self.pointing.az)
        ) / self.layer_depth

        if verbose:
            print(f"{(layer_wind_east, layer_wind_north) = }")

        # compute the offset with respect to the center of the scan
        center_az, center_el = utils.coords.get_center_lonlat(
            self.pointing.az, self.pointing.el
        )
        self.pointing.dx, self.pointing.dy = utils.coords.lonlat_to_xy(
            self.pointing.az, self.pointing.el, center_az, center_el
        )

        # the angular position of each detector over time WRT the atmosphere
        # this has dimensions (det index, time index)
        self.boresight_angular_position = np.c_[
            self.pointing.dx
            + np.cumsum(angular_velocity_x * self.pointing.dt, axis=-1),
            self.pointing.dy
            + np.cumsum(angular_velocity_y * self.pointing.dt, axis=-1),
        ]

        # find the detector offsets which form a convex hull
        self.detector_offsets = np.c_[self.array.offset_x, self.array.offset_y]
        doch = sp.spatial.ConvexHull(self.detector_offsets)

        # pad each vertex in the convex hull by an appropriate amount
        unit_circle_angles = np.linspace(0, 2 * np.pi, 32 + 1)[:-1]
        unit_circle_offsets = np.c_[
            np.cos(unit_circle_angles), np.sin(unit_circle_angles)
        ]
        angular_padding_per_detector = 1.1 * self.angular_beam_fwhm
        padded_doch_offsets = (
            angular_padding_per_detector[doch.vertices][:, None, None]
            * unit_circle_offsets[None, :]
            + self.detector_offsets[doch.vertices][:, None]
        ).reshape(-1, 2)

        # take the convex hull of those padded vertices
        padded_doch = sp.spatial.ConvexHull(padded_doch_offsets)
        detector_padding = padded_doch_offsets[padded_doch.vertices]

        # add the padded hull to the moving boresight
        atmosphere_offsets = (
            detector_padding[:, None] + self.boresight_angular_position[None]
        ).reshape(-1, 2)
        atmosphere_hull = sp.spatial.ConvexHull(atmosphere_offsets)
        self.atmosphere_hull_points = atmosphere_offsets[atmosphere_hull.vertices]

        # R takes us from the real (dx, dy) to a more compact (cross_section, extrusion) frame
        self.res = utils.linalg.optimize_area_minimizing_rotation_matrix(
            self.atmosphere_hull_points
        )

        assert self.res.success
        self.R = utils.linalg.get_rotation_matrix_2d(self.res.x[0])

        #
        #          ^      xxxxxxxxxxxx
        #          |      xxxxxxxxxxxx
        #   cross-section xxxxxxxxxxxx
        #          |      xxxxxxxxxxxx
        #          @      xxxxxxxxxxxx
        #

        trans_points = self.atmosphere_hull_points @ self.R.T
        cross_section_min = trans_points[:, 0].min()
        cross_section_max = trans_points[:, 0].max()
        extrusion_min = trans_points[:, 1].min()
        extrusion_max = trans_points[:, 1].max()

        self.cross_section_side = np.arange(
            cross_section_min,
            cross_section_max + self.angular_resolution,
            self.angular_resolution,
        )
        self.extrusion_side = np.arange(
            extrusion_min, extrusion_max, self.angular_resolution
        )

        self.n_cross_section = len(self.cross_section_side)
        self.n_extrusion = len(self.extrusion_side)

        CROSS_SECTION, EXTRUSION = np.meshgrid(
            self.cross_section_side, self.extrusion_side
        )

        self.TRANS_POINTS = np.concatenate(
            [CROSS_SECTION[..., None], EXTRUSION[..., None]], axis=-1
        )

        extrusion_indices = [
            0,
            *(2 ** np.arange(0, np.log(self.n_extrusion) / np.log(2))).astype(int),
            self.n_extrusion - 1,
        ]

        extrusion_sample_index = []
        cross_section_sample_index = []
        for i, extrusion_index in enumerate(extrusion_indices):
            n_ribbon_samples = np.minimum(
                np.maximum(
                    int(self.n_cross_section * RIBBON_SAMPLE_DECAY**-i),
                    MIN_SAMPLES_PER_RIBBON,
                ),
                self.n_cross_section,
            )
            cross_section_indices = np.unique(
                np.linspace(0, self.n_cross_section - 1, n_ribbon_samples).astype(int)
            )
            cross_section_sample_index.extend(cross_section_indices)
            extrusion_sample_index.extend(
                np.repeat(extrusion_index, len(cross_section_indices))
            )

        self.extrusion_sample_index = np.array(extrusion_sample_index)
        self.cross_section_sample_index = np.array(cross_section_sample_index)

        live_edge_positions = np.c_[
            self.cross_section_side,
            np.repeat(extrusion_min - self.angular_resolution, self.n_cross_section),
        ]

        sample_positions = np.c_[
            CROSS_SECTION[self.extrusion_sample_index, self.cross_section_sample_index],
            EXTRUSION[self.extrusion_sample_index, self.cross_section_sample_index],
        ]

        # the sampling index will look something like:
        #
        #          x111010000000001 ...
        #          x100000000000000 ...
        #          x110000000000000 ...
        #          x100000000000000 ...
        #  leading x111000000000001 ...
        #  edge    x100000000000000 ...
        #          x110000000000000 ...
        #          x100000000000000 ...
        #          x111000000000001 ...
        #

        self.n_live_edge = len(live_edge_positions)
        self.n_sample = len(sample_positions)

        if verbose:
            print(f"{self.n_extrusion = }")
            print(f"{self.n_live_edge = }")
            print(f"{self.n_sample = }")

        outer_scale = 0.1

        # sample upper {i,j}
        i, j = np.triu_indices(self.n_sample, k=1)
        COV_S_S = np.eye(self.n_sample) + JITTER_LEVEL
        COV_S_S[i, j] = matern_callback(
            np.sqrt(np.square(sample_positions[j] - sample_positions[i]).sum(axis=1))
            / outer_scale,
            5 / 6,
            n_test_points=256,
        )
        COV_S_S[j, i] = COV_S_S[i, j]

        # this one is explicit
        COV_LE_S = matern_callback(
            np.sqrt(
                np.square(sample_positions[None] - live_edge_positions[:, None]).sum(
                    axis=2
                )
            )
            / outer_scale,
            5 / 6,
            n_test_points=256,
        )

        # live edge upper {i,j}
        i, j = np.triu_indices(self.n_live_edge, k=1)
        COV_LE_LE = np.eye(self.n_live_edge) + JITTER_LEVEL
        COV_LE_LE[i, j] = matern_callback(
            np.sqrt(
                np.square(live_edge_positions[j] - live_edge_positions[i]).sum(axis=1)
            )
            / outer_scale,
            5 / 6,
            n_test_points=256,
        )
        COV_LE_LE[j, i] = COV_LE_LE[i, j]

        # this is typically the bottleneck
        inv_COV_S_S = utils.linalg.fast_psd_inverse(COV_S_S)

        # compute the weights
        self.A = COV_LE_S @ inv_COV_S_S
        self.B = np.linalg.cholesky(COV_LE_LE - self.A @ COV_LE_S.T)
        self.shaped_values = np.zeros(
            (self.n_extrusion, self.n_cross_section), dtype=np.float32
        )

    def simulate_normalized_effective_water_vapor(self):
        n_steps = self.n_extrusion

        extruded_values = extrude(
            values=self.shaped_values.ravel(),
            A=self.A,
            B=self.B,
            n_steps=n_steps,
            n_i=self.n_extrusion,
            n_j=self.n_cross_section,
            i_sample_index=self.extrusion_sample_index,
            j_sample_index=self.cross_section_sample_index,
        )

        self.shaped_values = extruded_values.reshape(
            self.n_extrusion, self.n_cross_section
        )

        trans_detector_angular_positions = (
            self.detector_offsets[:, None] + self.boresight_angular_position[None]
        ) @ self.R.T
        detector_values = np.zeros(trans_detector_angular_positions.shape[:-1])
        for uband in self.array.ubands:
            band_mask = self.array.dets.band == uband
            band_angular_fwhm = self.angular_beam_fwhm[band_mask].mean()
            F = utils.beam.make_beam_filter(
                band_angular_fwhm, self.angular_resolution, self.array.beam_profile
            )
            FILTERED_VALUES = utils.beam.separably_filter(self.shaped_values, F)
            detector_values[band_mask] = sp.interpolate.RegularGridInterpolator(
                (self.cross_section_side, self.extrusion_side), FILTERED_VALUES.T
            )(trans_detector_angular_positions[band_mask])

        return detector_values

    def simulate_integrated_water_vapor(self):
        detector_values = self.simulate_normalized_effective_water_vapor()

        # this is "zenith-scaled"
        self.line_of_sight_pwv = (
            self.weather.pwv
            * (1.0 + self.site.pwv_rms_frac * detector_values)
            / np.sin(self.EL)
        )
