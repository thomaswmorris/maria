from __future__ import annotations

import logging
from typing import Iterable

import arrow
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from ..array import Array
from ..band import Band, get_band
from ..coords import Coordinates
from ..instrument import Instrument, get_instrument
from ..site import Site, get_site
from .processing import process_tod  # noqa
from .tod import TOD, VALID_TOD_QUANTITIES  # noqa

TOD.process = process_tod  # to avoid circular imports


logger = logging.getLogger("maria")


def load(path: str, site: Site | str, bands: Iterable[Band | str], index: int = 1):

    bands = [band if isinstance(band, Band) else get_band(band) for band in bands]

    if isinstance(site, str):
        site = get_site(site)

    with fits.open(path) as f:
        raw = f[index].data

        det_uids, det_counts = np.unique(raw["PIXID"], return_counts=True)

        if det_counts.std() > 0:
            raise ValueError("Cannot reshape a ragged TOD.")

        n_dets = len(det_uids)
        n_samp = det_counts.max()

        data = {"data": raw["FNU"].astype("float32").reshape((n_dets, n_samp))}

        ra = raw["dx"].astype(float).reshape((n_dets, n_samp))  # rad
        dec = raw["dy"].astype(float).reshape((n_dets, n_samp))

        t = raw["time"].astype(float).reshape((n_dets, n_samp)).mean(axis=0)

        if "JDSTART" in f[index].header:
            t += Time(f[index].header["JDSTART"], format="jd").unix
        else:
            start_time = arrow.get()
            logger.warning(f"No start time specified, assuming start_time={start_time}")
            t += start_time.timestamp()

        boresight = Coordinates(
            t=t,
            phi=ra,
            theta=dec,
            earth_location=site.earth_location,
            frame="ra/dec",
        )

        # building array class
        dets_dict = {
            "xi": ra[:, 0] - ra[:, 0].mean(),  # in ra/dec frame
            "eta": dec[:, 0] - dec[:, 0].mean(),  # in ra/dec frame
            "band_name": len(dec[:, 0]) * ["m2/f093"],
        }

        dets_df = pd.DataFrame(dets_dict)
        _array = get_instrument("mustang-2").arrays[0]

        for col in _array.dets.columns:
            if col in dets_df.columns:
                continue
            dets_df[col] = _array.dets.iloc[0][col]

        dets = Array(name="dets", dets=dets_df, bands=bands)

        metadata = {
            "atmosphere": False,
            "altitude": float(site.altitude.m),
            "region": site.region,
        }

        tod = TOD(
            data=data,
            dets=dets,
            coords=boresight,
            units="K_RJ",
            metadata=metadata,
        )

        return tod
