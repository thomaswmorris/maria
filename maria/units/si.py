from __future__ import annotations

import os
import numpy as np
import pandas as pd

here, this_filename = os.path.split(__file__)

prefixes = pd.read_csv(f"{here}/si-prefixes.csv", index_col=0)
prefixes.loc[""] = "", "", 1e0
prefixes.sort_values("factor", ascending=True, inplace=True)
prefixes.loc[:, "time"] = np.log10(prefixes.factor.values) % 3 == 0
