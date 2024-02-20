import os
from dataclasses import dataclass, fields
from operator import attrgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from .. import utils
from .bands import BandList  # noqa F401
from .beams import compute_angular_fwhm, construct_beam_filter  # noqa F401
from .dets import Detectors  # noqa F401

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

# better formatting for pandas dataframes
# pd.set_eng_float_format()

here, this_filename = os.path.split(__file__)

# all_instrument_params = utils.io.read_yaml(f"{here}/../configs/default_params.yml")["instrument"]

INSTRUMENT_CONFIGS = utils.io.read_yaml(f"{here}/instruments.yml")

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
    for key, config in INSTRUMENT_CONFIGS.items():
        if instrument_name in config.get("aliases", []):
            instrument_name = key
    if instrument_name not in INSTRUMENT_CONFIGS.keys():
        raise InvalidInstrumentError(instrument_name)
    instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
    instrument_config.update(kwargs)
    return Instrument.from_config(instrument_config)


@dataclass
class Instrument:
    """
    An instrument.
    """

    description: str = ""
    primary_size: float = 5  # in meters
    field_of_view: float = 1  # in deg
    geometry: str = "hex"
    baseline: float = 0
    bath_temp: float = 0
    bands: BandList = None
    dets: pd.DataFrame = None  # dets, it's complicated
    documentation: str = ""
    vel_limit: float = 5  # in deg/s
    acc_limit: float = 2  # in deg/s^2

    @classmethod
    def from_config(cls, config):
        dets = Detectors.from_config(config=config)

        for key in ["dets", "aliases"]:
            if key in config:
                config.pop(key)

        return cls(bands=dets.bands, dets=dets, **config)

    def __repr__(self):
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self)) for f in fields(self) if f.name != "dets"
        )

        nodef_f_repr = []
        for name, value in nodef_f_vals:
            if name == "bands":
                nodef_f_repr.append(f"{name}={value.__short_repr__()}")
            else:
                nodef_f_repr.append(f"{name}={value}")

        return f"{self.__class__.__name__}({', '.join(nodef_f_repr)})"

    @property
    def ubands(self):
        return self.dets.ubands

    @property
    def offset_x(self):
        return self.dets.offset_x

    @property
    def offset_y(self):
        return self.dets.offset_y

    @property
    def offsets(self):
        return np.c_[self.offset_x, self.offset_y]

    @property
    def baseline_x(self):
        return self.dets.baseline_x

    @property
    def baseline_y(self):
        return self.dets.baseline_y

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
        return compute_angular_fwhm(z=z, fwhm_0=self.primary_size, n=1, f=1e9 * nu)

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

    def plot_dets(self, units="deg"):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=160)

        legend_handles = []
        for iub, uband in enumerate(self.ubands):
            band_mask = self.dets.band == uband

            fwhm = np.degrees(self.fwhm[band_mask])
            offsets = np.degrees(self.offsets[band_mask])

            if units == "arcmin":
                fwhm *= 60
                offsets *= 60

            if units == "arcsec":
                fwhm *= 60
                offsets *= 3600

            band_color = HEX_CODE_LIST[iub]

            # band_center = self.dets.band_center[band_mask].mean()
            # band_res_arcmins = 2 * self.fwhm

            # 60 * np.degrees(
            #     1.22 * 2.998e8 / (1e9 * band_center * self.primary_size)
            # )

            # offsets_arcmins = 60 * np.degrees(self.offsets[band_mask])

            ax.add_collection(
                EllipseCollection(
                    widths=fwhm,
                    heights=fwhm,
                    angles=0,
                    units="xy",
                    facecolors=band_color,
                    edgecolors="k",
                    lw=1e-1,
                    alpha=0.5,
                    offsets=offsets,
                    transOffset=ax.transData,
                )
            )

            legend_handles.append(
                Patch(
                    label=f"{uband}, res = {fwhm.mean():.03f} {units}",
                    color=band_color,
                )
            )

            ax.scatter(*offsets.T, label=uband, s=5e-1, color=band_color)

        ax.set_xlabel(rf"$\theta_x$ offset ({units})")
        ax.set_ylabel(rf"$\theta_y$ offset ({units})")
        ax.legend(handles=legend_handles)


instrument_data = pd.DataFrame(INSTRUMENT_CONFIGS).reindex(INSTRUMENT_DISPLAY_COLUMNS).T

for instrument_name, config in INSTRUMENT_CONFIGS.items():
    instrument = get_instrument(instrument_name)
    f_list = sorted(np.unique([band.center for band in instrument.dets.bands]))
    instrument_data.at[instrument_name, "f [GHz]"] = "/".join([str(f) for f in f_list])
    instrument_data.at[instrument_name, "n"] = instrument.dets.n

all_instruments = list(instrument_data.index)


class InvalidInstrumentError(Exception):
    def __init__(self, invalid_instrument):
        super().__init__(
            f"The instrument '{invalid_instrument}' is not supported. "
            f"Supported instruments are:\n\n{instrument_data.__repr__()}"
        )
