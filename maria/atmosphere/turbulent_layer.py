import dask.array as da
import numpy as np
import scipy as sp

from .. import utils
from ..coords import Coordinates, get_center_phi_theta
from ..functions import approximate_normalized_matern
from ..instrument import Instrument
from ..instrument.beam import construct_beam_filter, separably_filter_2d
from ..weather import Weather

MIN_SAMPLES_PER_RIBBON = 2
RIBBON_SAMPLE_DECAY = 2
JITTER_LEVEL = 1e-4

matern_callback = approximate_normalized_matern


class TurbulentLayer:
    """
    The simplest possible atmospheric model.
    This model is only appropriate for single instruments, i.e. when the baseline is zero.
    """

    def __init__(
        self,
        instrument: Instrument,
        boresight: Coordinates,
        weather: Weather,
        depth: float,
        res: float,
        turbulent_outer_scale: float = 500,
        verbose: bool = False,
        timestep: float = 0.05,
        **kwargs,
    ):
        self.instrument = instrument
        self.boresight = boresight.downsample(timestep=timestep)
        self.weather = weather
        self.depth = depth
        self.res = res
        self.timestep = timestep

        self.sim_time = self.boresight.time
        self.sim_az = self.boresight.az.compute()
        self.sim_el = self.boresight.el.compute()

        # this is approximately correct
        # self.depth = self.depth / np.sin(np.mean(self.plan.el))

        # this might change
        self.angular_outer_scale = turbulent_outer_scale / self.depth

        # returns the beam fwhm for each detector at the layer distance
        # self.physical_beam_fwhm = self.instrument.physical_fwhm(self.depth)
        # self.angular_beam_fwhm = self.physical_beam_fwhm / self.depth

        self.angular_resolution = self.res / self.depth

        if verbose:
            print(f"{self.angular_resolution=}")

        self.layer_altitude = self.weather.altitude + self.depth / np.sin(self.sim_el)

        layer_wind_north = sp.interpolate.interp1d(
            self.weather.altitude, self.weather.wind_north, axis=0
        )(self.layer_altitude)
        layer_wind_east = sp.interpolate.interp1d(
            self.weather.altitude, self.weather.wind_east, axis=0
        )(self.layer_altitude)

        angular_velocity_x = (
            +layer_wind_east * np.cos(self.sim_az)
            - layer_wind_north * np.sin(self.sim_az)
        ) / self.depth

        angular_velocity_y = (
            -layer_wind_east * np.sin(self.sim_az)
            + layer_wind_north * np.cos(self.sim_az)
        ) / self.depth

        if verbose:
            print(f"{(layer_wind_east, layer_wind_north)=}")

        # compute the offset with respect to the center of the scan
        center_az, center_el = get_center_phi_theta(self.sim_az, self.sim_el)
        dx, dy = self.boresight.offsets(frame="az_el", center=(center_az, center_el))

        # the angular position of the boresight over time WRT the atmosphere
        # this has dimensions (2, time index)
        self.boresight_angular_position = np.c_[
            dx + np.cumsum(angular_velocity_x * np.gradient(self.sim_time)),
            dy + np.cumsum(angular_velocity_y * np.gradient(self.sim_time)),
        ]

        # find the detector offsets which form a convex hull
        self.detector_offsets = np.radians(
            np.c_[self.instrument.sky_x, self.instrument.sky_y]
        )

        # add a small circle of offsets to account for the beams
        unit_circle_complex = np.exp(1j * np.linspace(0, 2 * np.pi, 64 + 1)[:-1])
        unit_circle_offsets = np.c_[
            np.real(unit_circle_complex), np.imag(unit_circle_complex)
        ]

        # this is a convex hull for the instrument if it's staring
        stare_convex_hull = sp.spatial.ConvexHull(
            (
                self.detector_offsets[None, :, None]
                + self.instrument.angular_fwhm(depth)[:, None, None]
                * unit_circle_offsets[None]
            ).reshape(-1, 2)
        )
        stare_convex_hull_points = stare_convex_hull.points[
            stare_convex_hull.vertices
        ].reshape(-1, 2)

        # convex hull downsample index, to get to 1 second
        chds_index = [
            *np.arange(
                0,
                len(self.sim_time),
                int(np.maximum(np.median(np.diff(self.sim_time)), 1)),
            ),
            -1,
        ]

        # this is a convex hull for the atmosphere
        atmosphere_hull = sp.spatial.ConvexHull(
            (
                self.boresight_angular_position[chds_index, None]
                + stare_convex_hull_points[None]
            ).reshape(-1, 2)
        )
        self.atmosphere_hull_points = atmosphere_hull.points.reshape(-1, 2)[
            atmosphere_hull.vertices
        ]

        # R takes us from the real (dx, dy) to a more compact (cross_section, extrusion) frame
        self.optres = utils.linalg.optimize_area_minimizing_rotation_matrix(
            self.atmosphere_hull_points
        )

        assert self.optres.success
        self.R = utils.linalg.get_rotation_matrix_2d(self.optres.x[0])

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
            print(f"{self.n_extrusion=}")
            print(f"{self.n_live_edge=}")
            print(f"{self.n_sample=}")

        # sample upper {i,j}
        i, j = np.triu_indices(self.n_sample, k=1)
        COV_S_S = np.eye(self.n_sample) + JITTER_LEVEL
        COV_S_S[i, j] = matern_callback(
            np.sqrt(np.square(sample_positions[j] - sample_positions[i]).sum(axis=1))
            / self.angular_outer_scale,
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
            / self.angular_outer_scale,
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
            / self.angular_outer_scale,
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

        self.shaped_values = da.from_array(self.shaped_values)

        self.atmosphere_detector_points = (
            self.detector_offsets[:, None] + self.boresight_angular_position[None]
        ) @ self.R.T

    def generate(self):
        n_steps = self.n_extrusion

        extruded_values = utils.linalg.extrude(
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

    def sample(self):
        detector_values = np.zeros(self.atmosphere_detector_points.shape[:-1])
        for band in self.instrument.bands:
            # we assume the atmosphere looks the same for every nu in the band

            band_index = self.instrument.dets.subset(band_name=band.name).index
            band_angular_fwhm = self.instrument.angular_fwhm(z=self.depth)[
                band_index
            ].mean()

            F = construct_beam_filter(
                fwhm=band_angular_fwhm,
                res=self.angular_resolution,
                beam_profile=self.instrument.beam_profile,
            )
            FILTERED_VALUES = separably_filter_2d(self.shaped_values, F)
            FILTERED_VALUES = self.shaped_values

            detector_values[band_index] = sp.interpolate.RegularGridInterpolator(
                (self.cross_section_side, self.extrusion_side), FILTERED_VALUES.T
            )(self.atmosphere_detector_points[band_index])

        return detector_values

    # def simulate_integrated_water_vapor(self):
    #     detector_values = self.simulate_normalized_effective_water_vapor()

    #     # this is "zenith-scaled"
    #     self.line_of_sight_pwv = (
    #         self.weather.pwv
    #         * (1.0 + self.pwv_rms_frac * detector_values)
    #         / np.sin(self.EL)
    #     )
