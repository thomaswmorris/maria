import numpy as np
from astropy.io import fits

from ... import TOD
from ...coords import Coordinates
from .. import instrument, site


def load_mustang2_tod(fname: str, hdu: int = 1):
    f = fits.open(fname)
    raw = f[hdu].data

    det_uids, det_counts = np.unique(raw["PIXID"], return_counts=True)

    if det_counts.std() > 0:
        raise ValueError("Cannot reshape a ragged TOD.")

    n_dets = len(det_uids)
    n_samp = det_counts.max()

    data = {"data": raw["FNU"].astype("float32").reshape((n_dets, n_samp))}

    ra = raw["dx"].astype(float).reshape((n_dets, n_samp))
    dec = raw["dy"].astype(float).reshape((n_dets, n_samp))
    t = 1.6e9 + raw["time"].astype(float).reshape((n_dets, n_samp)).mean(axis=0)

    coords = Coordinates(
        time=t,
        phi=ra,
        theta=dec,
        earth_location=site.get_location("green_bank"),
        frame="ra_dec",
    )

    m2_config = instrument.get_instrument_config(instrument_name="MUSTANG-2")
    m2_config["array"]["n"] = n_dets

    m2 = instrument.get_instrument(**m2_config)

    return TOD(coords=coords, dets=m2.dets, data=data, units={"data": "K"})
