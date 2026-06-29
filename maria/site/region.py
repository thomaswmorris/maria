import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .height_map import get_height_map

here, this_filename = os.path.split(__file__)

REGION_DISPLAY_COLUMNS = ["location", "country", "latitude", "longitude", "timezone"]
REGIONS = pd.read_csv(f"{here}/regions.csv", index_col=0)
all_regions = list(REGIONS.index.values)

region_data = REGIONS.loc[:, REGION_DISPLAY_COLUMNS]


class InvalidRegionError(Exception):
    def __init__(self, invalid_region):
        super().__init__(
            f"The region '{invalid_region}' is not supported. Supported regions are:\n\n{region_data.to_string()}",
        )


def plot_all_regions():

    height_map = get_height_map()

    moll_proj = hp.projector.CartesianProj(rot=0, xsize=2000)
    moll_proj.set_flip("geo")

    vec2pix_func = lambda x, y, z: hp.pixelfunc.vec2pix(2048, x, y, z)  # noqa
    m = moll_proj.projmap(height_map, vec2pix_func)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=256)

    n_y, n_x = m.shape
    x_side = np.linspace(-180, 180, n_x)
    y_side = np.linspace(-90, 90, n_y)
    ax.pcolormesh(
        x_side, y_side, np.isfinite(m), cmap=LinearSegmentedColormap.from_list("my_gradient", ["skyblue", "beige"])
    )

    override_region_alignment = {
        "green_bank": ("right", "bottom"),
        "mount_graham": ("left", "top"),
        "owens_valley": ("right", "bottom"),
        "pico_veleta": ("left", "top"),
        "princeton": ("left", "top"),
        # 'san_agustin': ('left', 'center'),
        "south_pole": ("center", "bottom"),
    }

    for region_name, region in REGIONS.iterrows():
        ax.scatter(region.longitude, region.latitude, c="r", s=4)

        ha, va = override_region_alignment.get(region_name, ("left", "bottom"))

        ax.annotate(
            xy=(region.longitude, region.latitude),
            fontsize=5,
            text=region_name,
            ha=ha,
            va=va,
        )

    ax.set_axis_off()
    ax.set_rasterized(True)
