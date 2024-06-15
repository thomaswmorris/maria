import glob
import os
import pickle
import re
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


class TODError(BaseException):
    pass


tod_cuts_base = "/projects/ACT/yilung/depot/release_20211025/TODCuts"
tod_filepath_base = "/projects/ACT/managed/actpol_data"
tod_offsets_base = "/projects/ACT/mhasse/depots/shared/RelativeOffsets"


def get_scode(t):
    i = np.digitize(
        t,
        bins=[
            1397018846,
            1426572705,
            1456846608,
            1488443939,
            1521156461,
            1549177098,
            1580660694,
            1611466488,
        ],
    )
    return f"s{i+13}"


def get_tod_filepath(tod_id):
    filepaths = glob.glob(
        f"{tod_filepath_base}/season*/merlin/{tod_id[:5]}/{tod_id[:-5]}.zip"
    )
    return filepaths[0] if len(filepaths) > 0 else None


def get_tod_info(tod_filepath):
    return m2.scripting.get_tod_info({"filename": tod_filepath})


def get_calibration(tod_id, det_uid, scheme="iv"):
    pa, nom_freq = re.sub(rf".*.*\..*\.ar(.):f(.*)", r"PA\1 \2", tod_id).split()
    band = f"{pa}_f{nom_freq}"

    cals = None

    if scheme == "iv":
        tod_info = m2.scripting.get_tod_info({"filename": get_tod_filepath(tod_id)})
        cals = m2.scripting.get_calibration(
            [{"type": "iv", "source": "data"}], tod_info=tod_info, det_uid=det_uid
        ).get_property("cal")[1]
        cals *= tod_info.array_data["optical_sign"][det_uid]
        return pd.DataFrame(cals, index=det_uid, columns=["cal"])

    if scheme == "bs":
        import ast

        tod_filepath = glob.glob(
            f"/projects/ACT/managed/actpol_data/season*/merlin/{tod_id[:5]}/{tod_id[:-5]}.zip"
        )[0]
        tod_info = m2.scripting.get_tod_info({"filename": tod_filepath})
        pattern = f"/projects/ACT/yilung/depot/release_20211025/Calibration/{pa.lower()}_f{nom_freq}_*/{tod_id[:5]}/{tod_id[:-5]}.cal"
        fps = glob.glob(pattern)

        if not len(fps) > 0:
            return None

        d = {}
        with open(fps[0], "r") as f:
            c = f.read()
        for l in c.split("\n"):
            if len(l) > 0:
                k, v = l.split(" = ")
                d[k] = np.array(ast.literal_eval(v))

        bias_steps = pd.DataFrame(d, index=d["det_uid"])
        bias_steps.drop(columns=["det_uid"], inplace=True)

        return bias_steps  # .reindex(det_uid)

    if scheme == "local":
        year = datetime.fromtimestamp(int(tod_id.split(".")[0])).year
        pattern = f"/scratch/gpfs/tm12/act/calibration/mr3f_{pa.lower()}_f{band}_s*/*/{tod_id[:-5]}.cal"
        fps = glob.glob(pattern)

        if not len(fps) > 0:
            return None

        d = read_dict(fps[0])
        cal, uid = d["cal"], d["det_uid"]
        det_uid = np.array(sorted(list(set(det_uid) & set(uid))))
        cal = cal[np.isin(uid, det_uid)]
        return cal


def get_cuts(tod_id, det_uid, scheme):
    pa, nom_freq = re.sub(rf".*.*\..*\.ar(.):(.*)", r"PA\1 \2", tod_id).split()

    if scheme == "depot":
        pattern = (
            f"{tod_cuts_base}/{pa.lower()}_{nom_freq}_*/{tod_id[:5]}/{tod_id[:-5]}.cuts"
        )
        fps = glob.glob(pattern)
        if len(fps) > 0:
            return m2.tod.cuts.TODCuts.read_from_path(fps[0])

    return None


def get_flatfield(tod_id, det_uid, scheme):
    pa, nom_freq = re.sub(rf".*.*\..*\.ar(.):(.*)", r"PA\1 \2", tod_id).split()
    band = f"{pa}_{nom_freq}"

    if scheme == "atm":
        return pd.read_csv(
            f"/home/tm12/act/products/atmosphere/flatfields/ff_{band}.csv", index_col=0
        )  # .reindex(det_uid)

    if scheme == "dummy":
        flatfield = pd.DataFrame(index=det_uid)
        flatfield["ff"] = 1.0  # dummy flatfield
        return flatfield


def get_offsets(tod_id, det_uid, base=None):
    ar = re.sub(r".*(ar[0-9]).*", r"\1", tod_id)
    fps = glob.glob(f"{tod_offsets_base}/template_{ar}_*.txt")
    offset_df = pd.read_csv(fps[-1], delim_whitespace=True, index_col=0)
    offset_df.columns = np.roll(offset_df.columns, -1)

    return offset_df.loc[(offset_df.x0 != 0) & (offset_df.y0 != 0)]  # .reindex(det_uid)
