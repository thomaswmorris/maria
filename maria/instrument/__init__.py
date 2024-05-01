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
from ..coords import Angle
from .bands import BandList  # noqa F401
from .beams import compute_angular_fwhm, construct_beam_filter  # noqa F401
from .detectors import Detectors  # noqa F401

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
    if instrument_name:
        for key, config in INSTRUMENT_CONFIGS.items():
            if instrument_name in config.get("aliases", []):
                instrument_name = key
        if instrument_name not in INSTRUMENT_CONFIGS.keys():
            raise InvalidInstrumentError(instrument_name)
        instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
    else:
        instrument_config = {}
    instrument_config.update(kwargs)
    return Instrument.from_config(instrument_config)


@dataclass
class Instrument:
    """
    An instrument.
    """

    description: str = "An instrument."
    primary_size: float = None  # in meters
    field_of_view: float = None  # in deg
    baseline: float = None
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

    def __post_init__(self):
        self.field_of_view = np.round(np.degrees(self.dets.sky_x.ptp()), 3)
        if self.field_of_view < 0.5 / 60:
            self.units = "arcsec"
        elif self.field_of_view < 0.5:
            self.units = "arcmin"
        else:
            self.units = "degrees"

    def __repr__(self):
        nodef_f_vals = (
            (f.name, attrgetter(f.name)(self)) for f in fields(self) if f.name != "dets"
        )

        nodef_f_repr = []
        for name, value in nodef_f_vals:
            if name == "bands":
                nodef_f_repr.append(f"bands=[{', '.join(value.names)}]")
            else:
                nodef_f_repr.append(f"{name}={value}")

        return f"{self.__class__.__name__}({', '.join(nodef_f_repr)})"

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
                    label=f"{uband}, res = {getattr(fwhms, fwhms.units)[band_mask].mean():.01f} {fwhms.units}",
                    color=band_color,
                )
            )

            ax.scatter(
                *getattr(offsets, offsets.units)[band_mask].T,
                label=uband,
                s=5e-1,
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
