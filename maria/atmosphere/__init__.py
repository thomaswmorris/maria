import numpy as np
import scipy as sp
import os
from tqdm import tqdm
import h5py
import time as ttime
from .. import utils, weather

# how do we do the bands? this is a great question.
# because all practical telescope instrumentation assume a constant band

here, this_filename = os.path.split(__file__)

from .. import base, utils

class AtmosphericSpectrum:
    def __init__(self, filepath):
        """
        A dataclass to hold spectra as attributes
        """
        with h5py.File(filepath, "r") as f:

            self.nu             = f["side_nu_Hz"][:]
            self.side_elevation = f["side_elevation_deg"][:]
            self.side_los_pwv   = f["side_line_of_sight_pwv_mm"][:]
            self.trj            = f["temperature_rayleigh_jeans_K"][:]
            self.phase_delay    = f["phase_delay_um"][:]


class BaseAtmosphericSimulation(base.BaseSimulation):
    """
    The base class for modeling atmospheric fluctuations.

    The methods to simulate e.g. line-of-sight water and temeperature profiles should be implemented by
    classes which inherit from this one. 
    """
    def __init__(self, array, pointing, site, **kwargs):
        super().__init__(array, pointing, site, **kwargs)

        utils.validate_pointing(self.pointing.az, self.pointing.el)

        self.weather = weather.Weather(
            t=self.pointing.time.mean(),
            region=self.site.region,
            altitude=self.site.altitude,
            quantiles=self.site.weather_quantiles,
        )

        spectrum_filepath = f"{here}/spectra/{self.site.region}.h5"
        self.spectrum = AtmosphericSpectrum(filepath=spectrum_filepath) if os.path.exists(spectrum_filepath) else None

    @property
    def EL(self):
        return utils.xy_to_lonlat(
            self.array.offset_x[:, None],
            self.array.offset_y[:, None],
            self.pointing.az,
            self.pointing.el)[1]

    def simulate_integrated_water_vapor(self):
        raise NotImplementedError('Atmospheric simulations are not implemented in the base class!')

    def _run(self, units='K_RJ'):

        if units == 'K_RJ': # Kelvin Rayleigh-Jeans

            self.simulate_integrated_water_vapor() 
            self.data = np.empty((self.array.n_dets, self.pointing.n_time), dtype=np.float32)

            for iub, uband in enumerate(self.array.ubands):

                band_mask = self.array.dets.band == uband

                passband  = (np.abs(self.spectrum.nu - self.array.dets.band_center[band_mask].mean()) < 0.5 * self.array.dets.band_width[band_mask].mean()).astype(float)
                passband /= passband.sum()

                band_T_RJ_interpolator = sp.interpolate.RegularGridInterpolator((self.spectrum.side_los_pwv, 
                                                                                 self.spectrum.side_elevation),
                                                                                (self.spectrum.trj * passband).sum(axis=-1))

                self.data[band_mask] = band_T_RJ_interpolator((self.line_of_sight_water_vapor[band_mask], np.degrees(self.EL[band_mask])))

        if units == 'F_RJ': # Fahrenheit Rayleigh-Jeans 🇺🇸

            self.simulate_temperature(self, units='K_RJ')
            self.data = 1.8 * (self.data - 273.15) + 32


                    

DEFAULT_ATMOSPHERE_CONFIG = {
    "min_depth": 500,
    "max_depth": 3000,
    "n_layers": 4,
    "min_beam_res": 4,
}

def get_rotation_matrix_from_skew(x):
    S = x * np.array([[0, 1], [-1, 0]])
    R = sp.linalg.expm(S)
    return R
            
MIN_SAMPLES_PER_RIBBON = 2
JITTER_LEVEL = 1e-4


class SingleLayerSimulation(BaseAtmosphericSimulation):
    """
    The simplest possible atmospheric model.
    This model is only appropriate for single instruments, i.e. when the baseline is zero.
    """

    def __init__(self, array, pointing, site, layer_height=1e3, min_beam_res=4, verbose=False, **kwargs):
        super().__init__(array, pointing, site, **kwargs)

        self.min_beam_res = min_beam_res
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

        layer_wind_north = sp.interpolate.interp1d(self.weather.altitude_levels, self.weather.wind_north, axis=0)(layer_altitude)
        layer_wind_east  = sp.interpolate.interp1d(self.weather.altitude_levels, self.weather.wind_east, axis=0)(layer_altitude)
        
        angular_velocity_x = (+layer_wind_east * np.cos(self.pointing.az) 
                              -layer_wind_north * np.sin(self.pointing.az)) / self.layer_depth
        
        angular_velocity_y = (-layer_wind_east * np.sin(self.pointing.az) 
                              +layer_wind_north * np.cos(self.pointing.az)) / self.layer_depth

        if verbose:
            print(f"{(layer_wind_east, layer_wind_north) = }")
        
        # compute the offset with respect to the center of the scan
        center_az, center_el = utils.get_center_lonlat(self.pointing.az, self.pointing.el)
        self.pointing.dx, self.pointing.dy = utils.lonlat_to_xy(self.pointing.az, self.pointing.el, center_az, center_el)

        # the angular position of each detector over time WRT the atmosphere
        # this has dimensions (det index, time index)
        self.boresight_angular_position = np.c_[self.pointing.dx + np.cumsum(angular_velocity_x * self.pointing.dt, axis=-1),
                                                self.pointing.dy + np.cumsum(angular_velocity_y * self.pointing.dt, axis=-1)]

        # find the detector offsets which form a convex hull
        self.detector_offsets = np.c_[self.array.offset_x, self.array.offset_y]
        doch = sp.spatial.ConvexHull(self.detector_offsets)
        
        # pad each vertex in the convex hull by an appropriate amount
        unit_circle_angles = np.linspace(0, 2*np.pi, 32+1)[:-1]
        unit_circle_offsets = np.c_[np.cos(unit_circle_angles), np.sin(unit_circle_angles)]
        angular_padding_per_detector = 1.1 * self.angular_beam_fwhm
        padded_doch_offsets = (angular_padding_per_detector[doch.vertices][:, None, None] * unit_circle_offsets[None, :] 
                               + self.detector_offsets[doch.vertices][:, None]).reshape(-1, 2)
        
        # take the convex hull of those padded vertices
        padded_doch = sp.spatial.ConvexHull(padded_doch_offsets)
        detector_padding = padded_doch_offsets[padded_doch.vertices]
        
        # add the padded hull to the moving boresight
        atmosphere_offsets = (detector_padding[:, None] + self.boresight_angular_position[None]).reshape(-1, 2)
        atmosphere_hull = sp.spatial.ConvexHull(atmosphere_offsets)
        self.atmosphere_hull_points = atmosphere_offsets[atmosphere_hull.vertices]

        # R takes us from the real (dx, dy) to a more compact (cross_section, extrusion) frame
        self.res = utils.optimize_area_minimizing_rotation_matrix(self.atmosphere_hull_points)
        
        assert self.res.success
        self.R = utils.get_rotation_matrix_from_angle(self.res.x[0])

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
        
        self.cross_section_side = np.arange(cross_section_min, cross_section_max + self.angular_resolution, self.angular_resolution)
        self.extrusion_side = np.arange(extrusion_min, extrusion_max, self.angular_resolution)
        
        self.n_cross_section = len(self.cross_section_side)
        self.n_extrusion = len(self.extrusion_side)
        
        CROSS_SECTION, EXTRUSION = np.meshgrid(self.cross_section_side, self.extrusion_side)
        
        self.TRANS_POINTS = np.concatenate([CROSS_SECTION[..., None], EXTRUSION[..., None]], axis=-1)
        
        extrusion_indices = [0, *(2 ** np.arange(0, np.log(self.n_extrusion) / np.log(2))).astype(int), self.n_extrusion-1]
        
    
        extrusion_sample_index = []
        cross_section_sample_index = []
        for i, extrusion_index in enumerate(extrusion_indices):
        
            n_ribbon_samples = np.minimum(np.maximum(int(self.n_cross_section * 2 ** -i), MIN_SAMPLES_PER_RIBBON), self.n_cross_section)
            cross_section_indices = np.unique(np.linspace(0, self.n_cross_section - 1, n_ribbon_samples).astype(int))
            cross_section_sample_index.extend(cross_section_indices)
            extrusion_sample_index.extend(np.repeat(extrusion_index, len(cross_section_indices)))

        self.extrusion_sample_index = np.array(extrusion_sample_index)
        self.cross_section_sample_index = np.array(cross_section_sample_index)
        
        live_edge_positions = np.c_[self.cross_section_side, 
                                    np.repeat(extrusion_min - self.angular_resolution, self.n_cross_section)]
        
        sample_positions = np.c_[CROSS_SECTION[self.extrusion_sample_index, self.cross_section_sample_index], 
                                 EXTRUSION[self.extrusion_sample_index, self.cross_section_sample_index]]
        
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
        COV_S_S[i, j] = utils.functions.approximate_normalized_matern(np.sqrt(np.square(sample_positions[j] - sample_positions[i]).sum(axis=1)) / outer_scale, 5/6, n_test_points=256)
        COV_S_S[j, i] = COV_S_S[i, j]
        
        # this one is explicit
        COV_LE_S = utils.functions.approximate_normalized_matern(np.sqrt(np.square(sample_positions[None] - live_edge_positions[:, None]).sum(axis=2)) / outer_scale, 5/6, n_test_points=256)
        
        # live edge upper {i,j}
        i, j = np.triu_indices(self.n_live_edge, k=1)
        COV_LE_LE = np.eye(self.n_live_edge) + JITTER_LEVEL
        COV_LE_LE[i, j] = utils.functions.approximate_normalized_matern(np.sqrt(np.square(live_edge_positions[j] - live_edge_positions[i]).sum(axis=1)) / outer_scale, 5/6, n_test_points=256)
        COV_LE_LE[j, i] = COV_LE_LE[i, j]
        
        # this is typically the bottleneck
        inv_COV_S_S = utils.fast_psd_inverse(COV_S_S)
        
        # compute the weights
        self.A = COV_LE_S @ inv_COV_S_S
        self.B = np.linalg.cholesky(COV_LE_LE - self.A @ COV_LE_S.T)
        self.VALUES = np.zeros((self.n_extrusion, self.n_cross_section), dtype=np.float32)


    def extrude(self, n_steps, desc=None):
        # muy rapido
        BUFFER = np.zeros((self.n_extrusion + n_steps, self.n_cross_section), dtype=np.float32)
        BUFFER[self.n_extrusion:] = self.VALUES
        for buffer_index in tqdm(np.arange(n_steps)[::-1], desc=desc):
            new_values = self.A @ BUFFER[buffer_index + self.extrusion_sample_index + 1, self.cross_section_sample_index] + self.B @ np.random.standard_normal(size=self.B.shape[-1])
            BUFFER[buffer_index] = new_values
        self.VALUES = BUFFER[:self.n_extrusion]
        

    def simulate_integrated_water_vapor(self):

        self.sim_start = ttime.time()
        self.extrude(n_steps=self.n_extrusion, desc="Generating atmosphere")
        
        trans_detector_angular_positions = (self.detector_offsets[:, None] + self.boresight_angular_position[None]) @ self.R.T
        detector_values = np.zeros(trans_detector_angular_positions.shape[:-1])
        for uband in self.array.ubands:
            band_mask = self.array.dets.band == uband
            band_angular_fwhm = self.angular_beam_fwhm[band_mask].mean()
            F = utils.beam.make_beam_filter(band_angular_fwhm, self.angular_resolution, self.array.beam_profile)
            FILTERED_VALUES = utils.beam.separably_filter(self.VALUES, F) 
            detector_values[band_mask] = sp.interpolate.RegularGridInterpolator((self.cross_section_side, self.extrusion_side), 
                                                                                FILTERED_VALUES.T)(trans_detector_angular_positions[band_mask])

        # this is "zenith-scaled"
        self.line_of_sight_water_vapor = self.weather.pwv * (1. + self.site.pwv_rms_frac * detector_values) / np.sin(self.EL)