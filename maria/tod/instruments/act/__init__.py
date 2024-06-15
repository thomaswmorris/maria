import glob
import importlib
import os
import pickle
import re
import sys
import time as ttime
from datetime import datetime

import astropy as ap
import matplotlib as mpl
import moby2 as m2
import numpy as np
import pandas as pd
import pylab as plt
import pytz
import scipy as sp
from IPython.display import clear_output
from scipy import interpolate, ndimage, optimize, signal, stats

base, this_filename = os.path.split(__file__)
sys.path.append(base)


class TODError(Exception):
    pass


from . import della


def get_therm_label(tod_id):
    pa = f"PA{tod_id[-6]}"

    therm_labels = {
        "Tr2_Arr_AtCF_Ar1": ["PA1", "PA4"],
        "Tr2_Arr_AtCF_Ar2": ["PA2", "PA5"],
        "Tr2_Arr_AtCF_Ar3": ["PA3", "PA6", "PA7"],
    }

    for k, v in therm_labels.items():
        if pa in v:
            return k


def load_moby_tod(
    tod_id,
    cal_schemes=["local", "bs", "iv"],
    cut_schemes=["depot", "auto"],
    flatfield_schemes=["atm", "dummy"],
    cluster="della",
    verbose=False,
):
    if re.match(r".*.*\..*\.ar(.):f(.*)", tod_id) is None:
        raise TODError(f'"{tod_id}" is not the right format for a TOD id.')

    cluster_module = importlib.import_module(cluster)

    res = {"id": tod_id, "status": None, "timestamp": np.round(ttime.time(), 3)}

    mtod = None
    start_time = ttime.time()

    # there may not be a copy of this TOD in the places we are looking
    tod_filepath = cluster_module.get_tod_filepath(tod_id)
    if tod_filepath == None:
        raise TODError("tod file not found")

    pa, nom_freq = re.sub(rf".*.*\..*\.ar(.):f(.*)", r"PA\1 \2", tod_id).split()
    band = f"{pa}_f{nom_freq}"
    tod_info = m2.scripting.get_tod_info({"filename": tod_filepath})

    det_uid_set = set(
        tod_info.array_data["det_uid"][tod_info.array_data["nom_freq"] == int(nom_freq)]
    )

    cuts = None
    for cut_scheme in cut_schemes:
        cuts = cluster_module.get_cuts(
            tod_id, det_uid=np.array(sorted(list(det_uid_set))), scheme=cut_scheme
        )
        if cuts is not None:
            res["cut_scheme"] = cut_scheme
            if verbose:
                print(
                    f'got cuts with scheme "{cut_scheme}" (n_det = {len(det_uid_set)})'
                )
            break
    if cuts is None:
        raise TODError(f"Could not get cuts with schemes {cut_schemes}")

    calibration = None
    for cal_scheme in cal_schemes:
        calibration = cluster_module.get_calibration(
            tod_id, det_uid=np.array(sorted(list(det_uid_set))), scheme=cal_scheme
        )
        if calibration is not None:
            res["cal_scheme"] = cal_scheme
            det_uid_set &= set(calibration.index.values)
            if verbose:
                print(
                    f'got calibration with scheme "{cal_scheme}" (n_det = {len(det_uid_set)})'
                )
            break
    if calibration is None:
        raise TODError(f"Could not get calibration with schemes {cal_schemes}")

    flatfield = None
    for flatfield_scheme in flatfield_schemes:
        flatfield = cluster_module.get_flatfield(
            tod_id, det_uid=np.array(sorted(list(det_uid_set))), scheme=flatfield_scheme
        )
        if flatfield is not None:
            res["flatfield_scheme"] = flatfield_scheme
            det_uid_set &= set(flatfield.index.values)
            if verbose:
                print(
                    f'got flatfield with scheme "{flatfield_scheme}" (n_det = {len(det_uid_set)})'
                )
            break
    if flatfield is None:
        raise TODError(f"Could not get flatfield with schemes {flatfield_schemes}")

    offsets = cluster_module.get_offsets(
        tod_id, det_uid=np.array(sorted(list(det_uid_set)))
    )
    det_uid_set &= set(offsets.index.values)
    if verbose:
        print(f"got offsets (n_det = {len(det_uid_set)})")

    det_uid = np.array(sorted(list(det_uid_set)))

    mtod = m2.scripting.get_tod({"filename": tod_filepath}, det_uid=det_uid)

    mtod.det_uid = det_uid
    mtod.calibration = calibration.reindex(det_uid)
    mtod.cuts = np.array(cuts.cuts, dtype=object)[det_uid] if cuts is not None else None
    mtod.flatfield = flatfield.reindex(det_uid)
    mtod.offsets = offsets.reindex(det_uid)
    mtod.tod_id = tod_id

    # mtod.abscal = float(abscal.loc[abscal.id==tod_id].cal) if tod_id in abscal.id.values else None

    try:
        mtod.thermometer = mtod.get_hk(get_therm_label(tod_id))
    except:
        mtod.thermometer = mtod.get_hk(
            get_therm_label(tod_id), start=0, count=len(mtod.ctime) - 256
        )
        mtod.thermometer = np.append(
            mtod.thermometer, np.repeat(mtod.thermometer[-1], 256)
        )

    if mtod.thermometer[0] is None:
        mtod.thermometer = np.zeros(len(mtod.ctime))

    res["status"] = "ok"
    res["elapsed"] = np.round(ttime.time() - start_time, 3)

    return mtod, res
