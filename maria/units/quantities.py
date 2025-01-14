import os
import re
import yaml
import pandas as pd

from .si import prefixes

here, this_filename = os.path.split(__file__)

with open(f"{here}/quantities.yml", "r") as f:
    QUANTITIES = (
        pd.DataFrame(yaml.safe_load(f)).set_index("quantity", drop=False).fillna("")
    )

prefixes_phrase = "|".join(prefixes.index)
base_units_phrase = "|".join(QUANTITIES.base_unit.values)
units_pattern = re.compile(
    rf"^(?P<prefix>({prefixes_phrase}))(?P<base_unit>{base_units_phrase})$"
)  # noqa


def parse_units(u):
    match = units_pattern.search(u)
    if match is None:
        raise ValueError(
            f"Invalid units '{u}'. Valid units are a combination of an SI prefix "
            f"(one of {prefixes_phrase}) and a base unit (one of {base_units_phrase}).",
        )
    units = match.groupdict()
    for attr in ["quantity", "SI", "from", "to"]:
        units[attr] = QUANTITIES.set_index("base_unit", drop=False).loc[
            units["base_unit"], attr
        ]
    units["factor"] = prefixes.loc[
        units["prefix"]
    ].factor  # if units["prefix"] else 1e0
    return units
