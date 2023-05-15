import os
import numpy as np
import scipy as sp
from astropy.io import fits

from . import utils

MAPPERS = [
    "Standard Binning",
    "Simple Common mode"
]

class InvalidMapperError(Exception):
    def __init__(self, invalid_mapper):
        print(f"The mapper \'{invalid_mapper}\' is not in the database of default mappers. "
        f"Default mappers are:{MAPPERS}")

def get_mapper(mapper_name, sky_data, array, lam, im, he, fname):
    if not mapper_name in MAPPERS:
        raise InvalidMapperError(mapper_name)
    if mapper_name == 'Standard Binning':
        Mapmaker = Binning(sky_data, array, lam, im, he, fname)
    if mapper_name == 'Simple Common mode':
        Mapmaker = Common_mode(sky_data, array, lam, im, he, fname)
    return Mapmaker

class Binning:
    def __init__(self, sky_data, array, lam, im, he, fname):
        self.sky_data = sky_data.copy()
        self.array    = array.copy() 
        self.lam      = lam.copy()
        self.im       = im.copy()
        self.he       = he.copy()
        self.fname    = fname.copy()

        self._get_coordinates()
        self._filter_model()

    def _get_coordinates(self):
        map_res = np.deg2rad(self.sky_data["incell"])
        map_nx, map_ny = self.im[0].shape
        map_x = map_res * map_nx * np.linspace(-0.5, 0.5, map_nx)
        map_y = map_res * map_ny * np.linspace(-0.5, 0.5, map_ny)
        map_X, map_Y = np.meshgrid(map_x, map_y, indexing="ij")
        lam_x, lam_y = utils.to_xy(self.lam.elev, self.lam.azim, self.lam.elev.mean(), self.lam.azim.mean())

        x_bins = np.arange(map_X.min(), map_X.max(), 8 * map_res)
        y_bins = np.arange(map_Y.min(), map_Y.max(), 8 * map_res)

        self.map_x = map_x
        self.map_y = map_y
        self.map_X = map_X
        self.map_Y = map_Y
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.x_bins = x_bins
        self.y_bins = y_bins

    def _filter_model(self):
        #this makes the TODs.
        self.map_data = np.empty((len(np.unique(self.array.bands)),len(self.x_bins)-1, len(self.y_bins)-1))

        for i in range(len(np.unique(self.array.bands))):
            self.map_data[i] = sp.interpolate.RegularGridInterpolator(
                (self.map_x, self.map_y), self.im[i], bounds_error=False, fill_value=0
            )((self.lam_x, self.lam_y))

        self.tod = np.zeros_like(self.map_data)

    def _binner(self, tod):
        map = sp.stats.binned_statistic_2d(
            self.map_X.ravel(),
            self.map_Y.ravel(),
            tod,
            statistic="mean",
            bins=(self.x_bins, self.y_bins),
        )[0]
        return map

    def _make_sky(self,i):
        #this combines the the tods
        self.combined = self.map_data + self.tod
        
        #bins them to make a map per frequency
        mock_obs = self._binner(self, self.combined[i].ravel())
        self.mockobs[i] = mock_obs

        #converts units based on meta data
        if self.sky_data['units'] == 'Jy/pixel':
            self.mockobs[i]     *= utils.KbrightToJyPix(np.unique(self.array.bands)[i], self.sky_data['incell'], self.sky_data['incell'])

        return map

    def _get_sky(self):

        self.mockobs     = np.empty((len(np.unique(self.array.bands)),len(self.x_bins)-1, len(self.y_bins)-1))

        # should mask the correct detectors...
        for iub in range(len(np.unique(self.array.bands))):
            self._make_sky(iub)
    
    def _savesky(self, file_save):
        if not os.path.exists(file_save):
            os.mkdir(file_save)

        fits.writeto(
            file_save + "/" + self.fname.replace(".fits", ".fits").split("/")[-1],
            self.mockobs,
            header=self.he,
            overwrite=True,
        )

    def add_tod(self, tod):
        self.tod = tod

    def run(self, savename):
        self._get_sky()
        self._savesky(savename)





class Common_mode:
    def __init__(self) -> None:
        pass