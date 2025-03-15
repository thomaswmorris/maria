from __future__ import annotations

import os

import numpy as np
import pandas as pd

here, this_filename = os.path.split(__file__)

PREFIXES = pd.read_csv(f"{here}/prefixes.csv", index_col=0)
PREFIXES.loc[""] = "", "", "", 0, 1e0
PREFIXES.sort_values("factor", ascending=True, inplace=True)
PREFIXES.loc[:, "primary"] = np.log10(PREFIXES.factor.values) % 3 == 0
