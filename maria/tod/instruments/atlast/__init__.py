import numpy as np
from astropy.io import fits

from .... import site
from ....instrument.array import Detectors
from ... import TOD
from ...coords import Coordinates


def load_atlast_tod(
    fname: str, hdu: int = 1, band_center: int = 93, band_width: int = 52
):
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
        earth_location=site.get_location("llano_de_chajnantor"),
        frame="ra_dec",
    )

    dets = Detectors.generate(
        bands_config={
            "f093": {
                "n_dets": n_dets,
                "band_center": band_center,
                "band_width": band_width,
            }
        }
    )

    return TOD(coords=coords, dets=dets, data=data, units={"data": "K"})
