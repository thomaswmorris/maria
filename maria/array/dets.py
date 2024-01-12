from collections.abc import Mapping

import matplotlib as mpl
import numpy as np
import pandas as pd

from .band import Band

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap('Paired')(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

REQUIRED_DET_CONFIG_KEYS = ['n', 'band_center', 'band_width']

det_columns = {
    'band': 'str',
    'offset_x': 'float',
    'offset_y': 'float',
    'baseline_x': 'float',
    'baseline_y': 'float',
    'baseline_z': 'float',
    'pol_angle': 'float',
}


def generate_array_offsets(geometry, field_of_view, n):
    valid_array_types = ['flower', 'hex', 'square']

    if geometry == 'flower':
        phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
        dzs = np.zeros(n).astype(complex)
        for i in range(n):
            dzs[i] = np.sqrt((i / (n - 1)) * 2) * np.exp(1j * phi * i)
        od = np.abs(np.subtract.outer(dzs, dzs))
        dzs *= field_of_view / od.max()
        return np.c_[np.real(dzs), np.imag(dzs)]
    if geometry == 'hex':
        return generate_hex_offsets(n, field_of_view)
    if geometry == 'square':
        dxy_ = np.linspace(-field_of_view, field_of_view, int(np.ceil(np.sqrt(n)))) / (
            2 * np.sqrt(2)
        )
        DX, DY = np.meshgrid(dxy_, dxy_)
        return np.c_[DX.ravel()[:n], DY.ravel()[:n]]

    raise ValueError(
        'Please specify a valid array type. Valid array types are:\n'
        + '\n'.join(valid_array_types)
    )


def generate_hex_offsets(n, d):
    angles = np.linspace(0, 2 * np.pi, 6 + 1)[1:] + np.pi / 2
    zs = np.array([0])
    layer = 0
    while len(zs) < n:
        for angle in angles:
            for z in layer * np.exp(1j * angle) + np.arange(layer) * np.exp(
                1j * (angle + 2 * np.pi / 3)
            ):
                zs = np.append(zs, z)
        layer += 1
    zs -= zs.mean()
    zs *= 0.5 * d / np.abs(zs).max()

    return np.c_[np.real(np.array(zs[:n])), np.imag(np.array(zs[:n]))]


class Detectors:
    @classmethod
    def generate(
        cls,
        bands: Mapping,
        field_of_view: float = 1,
        geometry: str = 'hex',
        baseline: float = 0,
        offsets: np.array = None,
    ):
        ubands = {}
        det_params = {
            col: []
            for col in [
                'band',
                'offset_x',
                'offset_y',
                'baseline_x',
                'baseline_y',
                'baseline_z',
                'pol_angle',
            ]
        }

        for band_key, band_config in bands.items():
            if not all(key in band_config.keys() for key in REQUIRED_DET_CONFIG_KEYS):
                raise ValueError(f'Each band must have keys {REQUIRED_DET_CONFIG_KEYS}')

            band_name = band_config.get('band_name', band_key)

            n = band_config['n']

            det_params['band'].extend(n * [band_name])

            ubands[band_key] = Band(
                name=band_name,
                center=band_config['band_center'],
                width=band_config['band_width'],
            )

            det_offsets_radians = np.radians(
                generate_array_offsets(geometry, field_of_view, n)
            )

            # should we make another function for this?
            det_baselines_meters = generate_array_offsets(geometry, baseline, n)

            # if randomize_offsets:
            #     np.random.shuffle(offsets_radians)  # this is a stupid function.

            det_params['offset_x'].extend(det_offsets_radians[:, 0])
            det_params['offset_y'].extend(det_offsets_radians[:, 1])
            det_params['baseline_x'].extend(det_baselines_meters[:, 0])
            det_params['baseline_y'].extend(det_baselines_meters[:, 1])
            det_params['baseline_z'].extend(0 * det_baselines_meters[:, 1])

            det_params['pol_angle'].extend(n * [0.0])

        return cls(det_params, bands=ubands)

    @property
    def band_center(self):
        centers = np.zeros(shape=self.n)
        for band_name, band in self.bands.items():
            centers[self.band == band_name] = band.center

        return centers

    @property
    def band_width(self):
        widths = np.zeros(shape=self.n)
        for band_name, band in self.bands.items():
            widths[self.band == band_name] = band.width

        return widths

    def __init__(self, params: dict = {}, bands: dict = {}):
        self.bands = bands
        self.params = params

        for k, v in self.params.items():
            setattr(self, k, np.array(v))

        self.n = len(self.band)

    @property
    def df(self):
        df = pd.DataFrame(columns=det_columns, dtype='float')

        for col, dtype in det_columns.items():
            df.loc[:, col] = self.params[col]
            df.loc[:, col] = df.loc[:, col].astype(dtype)

        return df

    @property
    def ubands(self):
        return list(self.bands.keys())

    def passband(self, nu):
        _nu = np.atleast_1d(nu)

        PB = np.zeros((len(self.df), len(_nu)))

        for band_name, band in self.bands.items():
            PB[self.band == band_name] = band.passband(_nu)

        return PB
