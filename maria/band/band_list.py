from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .band import BAND_FIELD_FORMATS, Band, get_band


class BandList(Sequence):
    def __init__(self, bands: Mapping | list = []):
        self.bands = []

        if isinstance(bands, BandList):
            self.bands = bands.bands

        elif isinstance(bands, Mapping):
            for band_name, band_config in bands.items():
                self.bands.append(Band(name=band_name, **band_config))

        elif isinstance(bands, list):
            for band in bands:
                if isinstance(band, Band):
                    self.bands.append(band)
                elif isinstance(band, str):
                    self.bands.append(get_band(band))
                else:
                    self.bands.append(Band(**band))

    # @classmethod
    # def from_list(cls, bands):
    #     band_list = []
    #     for band in bands:
    #         if isinstance(band, str):
    #             band_list.append(Band(name=band, **BAND_CONFIGS[band]))
    #         elif isinstance(band, Band):
    #             band_list.append(band)
    #         else:
    #             raise ValueError("'band' must be either a Band or a string.")
    #     return cls(band_list)

    # @classmethod
    # def from_config(cls, config):
    #     bands = []

    #     if isinstance(config, str):
    #         config = read_yaml(f"{here}/{config}")

    #     for name in config.keys():
    #         band_config = config[name]
    #         if "file" in band_config.keys():
    #             band_config = read_yaml(f'{here}/{band_config["file"]}')

    #         bands.append(Band(name=name, **band_config))
    #     return cls(bands=bands)

    @property
    def nu_min(self):
        return min([b.nu.min() for b in self.bands])

    @property
    def nu_max(self):
        return max([b.nu.max() for b in self.bands])

    def plot(self):
        for band in self.bands:
            fig, ax = plt.subplots(1, 1)
            ax.plot(band.nu, band.tau, label=band.name)

        ax.set_xlabel(r"$\nu$ [GHz]")
        ax.set_ylabel(r"$\tau(\nu)$ [Rayleigh-Jeans]")
        ax.legend()

    def add(self, band):
        if not isinstance(band, Band):
            raise ValueError("'band' must be a Band type.")
        if band.name in self.names:
            raise RuntimeError(f"There is already a band called '{band.name}'.")
        self.bands.append(band)

    def __getattr__(self, attr):
        if attr in self.names:
            return self.__getitem__(attr)
        if all([hasattr(band, attr) for band in self.bands]):
            return [getattr(band, attr) for band in self.bands]
        raise AttributeError(f"'BandList' object has no attribute '{attr}'")

    def __getitem__(self, index):
        if type(index) is int:
            return self.bands[index]
        elif type(index) is str:
            if index not in self.names:
                raise ValueError(f"BandList has no band named {index}.")
            return self.bands[self.names.index(index)]
        else:
            raise ValueError(
                f"Invalid index {index}. A bandList must be indexed by either an integer or a string.",
            )

    def __len__(self):
        return len(self.bands)

    def __repr__(self):
        return self.summary().__repr__()

    def _repr_html_(self):
        return self.summary()._repr_html_()

    def __short_repr__(self):
        return f"BandList([{', '.join(self.names)}])"

    @property
    def names(self):
        return [band.name for band in self.bands]

    def summary(self) -> pd.DataFrame:
        return pd.concat([band.summary() for band in self.bands], axis=1).T
