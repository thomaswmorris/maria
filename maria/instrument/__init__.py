import copy
import os
from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from ..io import read_yaml
from ..units import Angle
from ..utils import lazy_diameter
from .bands import BAND_CONFIGS, Band, BandList, parse_bands  # noqa
from .beams import compute_angular_fwhm
from .detectors import Detectors, generate_array

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

# better formatting for pandas dataframes
# pd.set_eng_float_format()

here, this_filename = os.path.split(__file__)

INSTRUMENT_CONFIGS = read_yaml(f"{here}/configs.yml")

for name, config in INSTRUMENT_CONFIGS.items():
    config["aliases"] = config.get("aliases", [])
    config["aliases"].append(name.lower())

INSTRUMENT_DISPLAY_COLUMNS = [
    "description",
    # "field_of_view",
    # "primary_size",
    # "bands",
]


def get_instrument_config(instrument_name=None, **kwargs):
    if instrument_name not in INSTRUMENT_CONFIGS.keys():
        raise InvalidInstrumentError(instrument_name)
    instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
    return instrument_config


def get_instrument(instrument_name="default", **kwargs):
    """
    Get an instrument from a pre-defined config.
    """
    if instrument_name:
        for key, config in INSTRUMENT_CONFIGS.items():
            if instrument_name.lower() in config.get("aliases", []):
                instrument_name = key
        if instrument_name not in INSTRUMENT_CONFIGS.keys():
            raise InvalidInstrumentError(instrument_name)
        instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
    else:
        instrument_config = {}
    instrument_config.update(kwargs)
    return Instrument.from_config(instrument_config)


subarray_params_to_inherit = [
    "array_packing",
    "array_shape",
    "array_offset",
    "beam_spacing",
    "baseline_diameter",
    "baseline_packing",
    "baseline_shape",
    "baseline_offset",
    "bath_temp",
    "polarization",
    "primary_size",
    "bands",
    "field_of_view",
]

band_params_to_inherit = [
    "time_constant",  # seconds
    "white_noise",  # Kelvin
    "pink_noise",  # Kelvin / s
    "efficiency",
]

passband_params_to_inherit = {
    "center": 150,  # GHz
    "width": 30,  # GHz
    "shape": "top_hat",
}

allowed_subarray_params = {
    "n": "int",
    "array_packing": "float",
    "array_shape": "float",
    "array_offset": "float",
    "beam_spacing": "float",
    "baseline_diameter": "float",
    "baseline_packing": "float",
    "baseline_shape": "float",
    "baseline_offset": "float",
    "bath_temp": "float",
    "polarization": "float",
    "primary_size": "float",
    "field_of_view": "float",
    "bands": "float",
}


def check_subarray_format(subarray):
    if isinstance(subarray.get("file"), str):
        return True
    if isinstance(subarray, Mapping):
        return all(k in allowed_subarray_params for k in subarray)
    return False


def get_subarrays(instrument_config):
    """
    Make the subarrays!
    """
    config = copy.deepcopy(instrument_config)

    if ("array" in config) and ("subarrays" not in config):
        subarray = config.pop("array")
        if check_subarray_format(subarray):
            config["subarrays"] = {"array": subarray}
        else:
            raise ValueError(f"Invalid array configuration: {subarray}")

    subarrays = {}

    for subarray_name in config["subarrays"]:
        subarray = config["subarrays"][subarray_name]

        if "file" in subarray:  # it points to a file:
            if not os.path.exists(subarray["file"]):
                subarray["file"] = f"{here}/detectors/arrays/{subarray['file']}"
            df = pd.read_csv(subarray["file"], index_col=0)

            if "bands" not in subarray:
                subarray["bands"] = {}

            subarray["n"] = len(df)
            for band_name in np.unique(df.band_name.values):
                subarray["bands"][band_name] = Band(
                    name=band_name, **BAND_CONFIGS[band_name]
                )

        elif ("n" not in subarray) and ("field_of_view" not in subarray):
            raise ValueError(
                "You must specificy one of 'n' or 'field_of_view' to generate an array."
            )

        subarray["bands"] = BandList(parse_bands(subarray["bands"]))

        for param in subarray_params_to_inherit:
            if (param in config) and (param not in subarray):
                subarray[param] = config[param]

        # for band_name, band_config in subarray["bands"].items():
        #     for param in band_params_to_inherit:
        #         if param in config:
        #         band_config[param] = band_config.get(param, default_value)

        #     if "passband" not in band_config:
        #         for param, default_value in passband_params_to_inherit:
        #             band_config[param] = band_config.get(param, default_value)

        subarrays[subarray_name] = subarray

    return subarrays


class Instrument:
    """
    An instrument.
    """

    @classmethod
    def from_config(cls, config):
        subarrays = get_subarrays(copy.deepcopy(config))

        bands = BandList(bands=[])
        df = pd.DataFrame(columns=["uid", "array_name", "band_name", "band_center"])

        for subarray_name, subarray in subarrays.items():
            array_bands = subarray["bands"]

            array_df = generate_array(**subarray)

            if "file" in subarray:
                for col, values in pd.read_csv(
                    subarray["file"], index_col=0
                ).T.iterrows():
                    array_df[col] = values

            # add leading zeros to detector uids
            fill_level = int(np.log(np.maximum(len(array_df) - 1, 1)) / np.log(10) + 1)

            uid_predix = f"{subarray_name}_" if subarray_name else ""
            uids = [
                f"{uid_predix}{str(i).zfill(fill_level)}" for i in range(len(array_df))
            ]

            array_df.insert(0, "uid", uids)
            array_df.insert(1, "array_name", subarray_name)

            df = pd.concat([df, array_df])

            for band in array_bands:
                if band not in bands.bands:
                    bands.add(band)

        df.index = np.arange(len(df))

        dets = Detectors(df=df, bands=bands)

        for key in ["dets", "aliases"]:
            if key in config:
                config.pop(key)

        return cls(bands=dets.bands, dets=dets)

    def __init__(
        self,
        description: str = "An instrument.",
        primary_size: float = None,  # in meters
        field_of_view: float = None,  # in deg
        baseline: float = None,
        bands: BandList = None,
        dets: pd.DataFrame = None,  # dets, it's complicated
        documentation: str = "",
        vel_limit: float = 5,  # in deg/s
        acc_limit: float = 2,  # in deg/s^2
    ):
        """
        Parameters
        ----------
        vel_limit : type
            The maximum angular speed of the array.
        """

        self.description = description
        self.primary_size = primary_size
        self.documentation = documentation
        self.bands = bands
        self.dets = dets
        self.vel_limit = vel_limit
        self.acc_limit = acc_limit

        self.primary_size = float(self.dets.primary_size.max())
        self.field_of_view = np.round(np.degrees(lazy_diameter(self.dets.offsets)), 3)
        self.baseline = np.round(lazy_diameter(self.dets.baselines), 3)

        if self.field_of_view < 0.5 / 60:
            self.units = "arcsec"
        elif self.field_of_view < 0.5:
            self.units = "arcmin"
        else:
            self.units = "degrees"

    def __repr__(self):
        bands = ",".join(self.bands.name)
        return f"Instrument({self.dets}, primary_size={self.primary_size} m, bands=[{bands}])"

    @property
    def ubands(self):
        return self.dets.bands.names

    @property
    def sky_x(self):
        return self.dets.sky_x

    @property
    def sky_y(self):
        return self.dets.sky_y

    @property
    def offsets(self):
        return np.c_[self.sky_x, self.sky_y]

    @property
    def baseline_x(self):
        return self.dets.baseline_x

    @property
    def baseline_y(self):
        return self.dets.baseline_y

    @property
    def baseline_z(self):
        return self.dets.baseline_z

    @property
    def baselines(self):
        return np.c_[self.baseline_x, self.baseline_y, self.baseline_z]

    @staticmethod
    def beam_profile(r, fwhm):
        return np.exp(np.log(0.5) * np.abs(r / fwhm) ** 8)

    @property
    def n_dets(self):
        return self.dets.n

    @property
    def fwhm(self):
        """
        Returns the angular FWHM (in radians) at infinite distance.
        """
        return self.angular_fwhm(z=np.inf)

    def angular_fwhm(self, z):  # noqa F401
        """
        Angular beam width (in radians) as a function of depth (in meters)
        """
        nu = self.dets.band_center  # in GHz
        return compute_angular_fwhm(z=z, fwhm_0=self.dets.primary_size, n=1, f=1e9 * nu)

    def physical_fwhm(self, z):
        """
        Physical beam width (in meters) as a function of depth (in meters)
        """
        return z * self.angular_fwhm(z)

    # def angular_beam_filter(self, z, res, beam_profile=None, buffer=1):  # noqa F401
    #     """
    #     Angular beam width (in radians) as a function of depth (in meters)
    #     """
    #     return construct_beam_filter(self.angular_fwhm(z), res, beam_profile=beam_profile, buffer=buffer)

    # def physical_beam_filter(self, z, res, beam_profile=None, buffer=1):  # noqa F401
    #     """
    #     Angular beam width (in radians) as a function of depth (in meters)
    #     """
    #     return construct_beam_filter(self.physical_fwhm(z), res, beam_profile=beam_profile, buffer=buffer)

    def plot(self, units=None):
        HEX_CODE_LIST = [
            mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
            for t in [*np.linspace(0.05, 0.95, 12)]
        ]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=160)

        fwhms = Angle(self.fwhm)
        offsets = Angle(self.offsets)

        legend_handles = []
        for iub, uband in enumerate(self.ubands):
            band_mask = self.dets.band_name == uband

            band_color = HEX_CODE_LIST[iub]

            ax.add_collection(
                EllipseCollection(
                    widths=getattr(fwhms, offsets.units)[band_mask],
                    heights=getattr(fwhms, offsets.units)[band_mask],
                    angles=0,
                    units="xy",
                    facecolors=band_color,
                    edgecolors="k",
                    lw=1e-1,
                    alpha=0.5,
                    offsets=getattr(offsets, offsets.units)[band_mask],
                    transOffset=ax.transData,
                )
            )

            legend_handles.append(
                Patch(
                    label=f"{uband}, (n={band_mask.sum()}, "
                    f"res={getattr(fwhms, fwhms.units)[band_mask].mean():.01f} {fwhms.units})",
                    color=band_color,
                )
            )

            ax.scatter(
                *getattr(offsets, offsets.units)[band_mask].T,
                label=uband,
                s=0,
                color=band_color,
            )

        ax.set_xlabel(rf"$\theta_x$ offset ({offsets.units})")
        ax.set_ylabel(rf"$\theta_y$ offset ({offsets.units})")
        ax.legend(handles=legend_handles)

        xls, yls = ax.get_xlim(), ax.get_ylim()
        cen_x, cen_y = np.mean(xls), np.mean(yls)
        wid_x, wid_y = np.ptp(xls), np.ptp(yls)
        radius = 0.5 * np.maximum(wid_x, wid_y)

        ax.set_xlim(cen_x - radius, cen_x + radius)
        ax.set_ylim(cen_y - radius, cen_y + radius)


instrument_data = pd.DataFrame(INSTRUMENT_CONFIGS).reindex(INSTRUMENT_DISPLAY_COLUMNS).T

# for instrument_name, config in INSTRUMENT_CONFIGS.items():
#     instrument = get_instrument(instrument_name)
#     f_list = sorted(np.unique([band.center for band in instrument.dets.bands]))
#     instrument_data.at[instrument_name, "f [GHz]"] = "/".join([str(f) for f in f_list])
#     instrument_data.at[instrument_name, "n"] = instrument.dets.n

all_instruments = list(instrument_data.index)


class InvalidInstrumentError(Exception):
    def __init__(self, invalid_instrument):
        super().__init__(
            f"The instrument '{invalid_instrument}' is not supported. "
            f"Supported instruments are:\n\n{instrument_data.__repr__()}"
        )
