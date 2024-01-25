import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from operator import attrgetter
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Patch

from .. import utils
from .band import Band, BandList, generate_bands # noqa F401
from .dets import Detectors, generate_detectors # noqa F401
from .beam import compute_angular_fwhm, construct_beam_filter

HEX_CODE_LIST = [
    mpl.colors.to_hex(mpl.colormaps.get_cmap("Paired")(t))
    for t in [*np.linspace(0.05, 0.95, 12)]
]

# better formatting for pandas dataframes
# pd.set_eng_float_format()

here, this_filename = os.path.split(__file__)

#all_instrument_params = utils.io.read_yaml(f"{here}/../configs/default_params.yml")["instrument"]

INSTRUMENT_CONFIGS = utils.io.read_yaml(f"{here}/instruments.yml")

INSTRUMENT_DISPLAY_COLUMNS = ["instrument_description", "field_of_view", "primary_size", "bands"]
instrument_data = pd.DataFrame(INSTRUMENT_CONFIGS).reindex(INSTRUMENT_DISPLAY_COLUMNS).T

for instrument_name, config in INSTRUMENT_CONFIGS.items():
    instrument_data.at[instrument_name, "bands"] = list(config["bands"].keys())

all_instruments = list(instrument_data.index)


class InvalidInstrumentError(Exception):
    def __init__(self, invalid_instrument):
        super().__init__(
            f"The instrument '{invalid_instrument}' is not supported."
            f"Supported instruments are:\n\n{instrument_data.__repr__()}"
        )


def get_instrument_config(instrument_name=None, **kwargs):
    if instrument_name not in INSTRUMENT_CONFIGS.keys():
        raise InvalidInstrumentError(instrument_name)
    instrument_config = INSTRUMENT_CONFIGS[instrument_name].copy()
    return instrument_config


def get_instrument(instrument_name="default", **kwargs):
    """
    Get an instrument from a pre-defined config.
    """
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


    @classmethod
    def from_config(cls, config):

       
        if isinstance(config.get("bands"), Mapping):
            dets = Detectors.generate(
                bands_config=config.pop("bands"),
                field_of_view=config.get("field_of_view", 1),
                geometry=config.get("geometry", "hex"),
                baseline=config.get("baseline", 0),
            )

        else:
            raise ValueError("'bands' must be a dictionary of bands.")


        return cls(
            bands=dets.bands,
            dets=dets,
            **config
        )

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
        return np.c_[self.baseline_x, self.baseline_y]


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

            # nom_freq = self.dets.band_center[band_mask].mean()
            # band_res_arcmins = 2 * self.fwhm

            # 60 * np.degrees(
            #     1.22 * 2.998e8 / (1e9 * nom_freq * self.primary_size)
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
