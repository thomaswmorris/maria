import h5py

import numpy as np

def read_weather_quantile_data(path: str, fields: list[str] = None):

    qdata = {"metadata": {}, "levels": {}}
    with h5py.File(path, "r") as f:

        if fields is None:
            fields = list(f["levels"].keys())

        qdata["metadata"]["side_quantile"] = f["metadata"]["side_quantile"][:]
        qdata["metadata"]["side_year_day"] = f["metadata"]["side_year_day"][:]
        qdata["metadata"]["side_day_hour"] = f["metadata"]["side_day_hour"][:]
        qdata["metadata"]["side_pressure"] = f["metadata"]["side_pressure"][:]

        qdata["metadata"]["units"] = {}
        for key in fields:
            qdata["levels"][key] = f["levels"][key][:]
    
            if f["levels"][key].attrs["log"]:
                qdata["levels"][key] = np.exp(qdata["levels"][key])

            qdata["metadata"]["units"][key] = f["levels"][key].attrs["units"]

    return qdata