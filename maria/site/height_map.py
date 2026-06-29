import h5py
import numpy as np

from ..io import fetch


def get_height_map():
    with h5py.File(fetch("world_heightmap.h5"), "r") as f:
        height_map = f["data"][:].astype(np.uint16)
    return 32 * np.where(height_map < 255, height_map, np.nan)
