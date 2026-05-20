import logging
import os
import pathlib
import re

import arrow
import numpy as np
import pandas as pd
import yaml

from ..utils import compute_resolution_precision, deg_to_signed_dms, deg_to_signed_hms, round_sig_figs
from .prefixes import PREFIXES

here, this_filename = os.path.split(__file__)
logger = logging.getLogger("maria")

PREFIXES = pd.read_csv(f"{here}/prefixes.csv", index_col=0)
PREFIXES.loc[""] = "", "", "", 0, 1e0
PREFIXES.sort_values("factor", ascending=True, inplace=True)
PREFIXES.loc[:, "primary"] = np.log10(PREFIXES.factor.values) % 3 == 0

unit_entries = {}
quantity_entries = {}
dimension_vector_entries = {}


class UnitError(Exception): ...


for path in pathlib.Path(f"{here}/physical_quantities").glob("*.yml"):
    with open(path) as f:
        for quantity, q in yaml.safe_load(f).items():
            if "units" in q:
                q_units = q.pop("units")
                for unit, config in q_units.items():
                    config["physical_quantity"] = quantity
                    config["aliases"] = set([unit, config["long_name"], *config.get("aliases", [])])
                unit_entries.update(q_units)
            quantity_entries[quantity] = q
            dimension_vector_entries[quantity] = q["vector"]

UNITS = pd.DataFrame(unit_entries).T
QUANTITIES = pd.DataFrame(quantity_entries).T
QUANTITY_DIMENSION_VECTORS = pd.DataFrame(dimension_vector_entries).sort_index().T.fillna(0)

UNITS["units"] = UNITS.index
UNITS["human"] = UNITS["human"].astype(bool).fillna(True)
UNITS["symbol"] = UNITS["symbol"].fillna("")
UNITS["factor"] = UNITS["factor"].astype(float)
UNITS["lowered"] = [unit.lower() for unit in UNITS.index]
UNITS["composite"] = UNITS["composite"].fillna(False)

prefixes_phrase = r"|".join(PREFIXES.index)

base_units_list = [unit for unit, entry in UNITS.iterrows() if not entry.composite]

base_units_aliases = {}
for unit, entry in UNITS.iterrows():
    base_units_aliases[unit] = set()
    for alias in entry.aliases:
        base_units_aliases[unit] |= {alias}

base_units_phrase = r"|".join([alias for aliases in base_units_aliases.values() for alias in aliases]).replace("^", "\\^")
units_pattern_case_sensitive = re.compile(
    rf"^(?P<modifiers>([/√ ]*))(?P<prefix>({prefixes_phrase}))(?P<parsed_unit>{base_units_phrase})( *(\^|\*\*)? *(?P<power>[-\.\d]+))?$"  # noqa
)
units_pattern_case_insensitive = re.compile(
    rf"^(?P<modifiers>([/√ ]*))(?P<prefix>({prefixes_phrase}))(?P<parsed_unit>(?i:{base_units_phrase}))( *(\^|\*\*)? *(?P<power>[-\.\d]+))?$"  # noqa
)


def parse_units(units):

    factor = 1
    dimension_vector = pd.Series(0.0, index=QUANTITY_DIMENSION_VECTORS.columns, dtype=float)

    units_repr_parts = []
    math_name_parts = []

    subunits = [s.strip() for s in re.compile(r"(/?√? *[A-Za-z_]+[ \*\^\-\.\d]*)").findall(units)]

    logger.debug(f"Parsing units {units}")

    for subunit in subunits:
        match = units_pattern_case_sensitive.search(subunit)
        if match is None:
            match = units_pattern_case_insensitive.search(subunit)
            if match is None:
                raise UnitError(
                    f"Invalid subunit '{subunit.strip('/')}'. Valid units are a combination of an SI prefix "
                    f"(one of {prefixes_phrase}) and a base unit (one of {'|'.join(base_units_list)}).",
                )

        su = match.groupdict()
        su["power"] = float(su["power"]) if su["power"] else 1.0
        if "/" in su["modifiers"]:
            su["power"] *= -1
        if "√" in su["modifiers"]:
            su["power"] *= 0.5

        logger.debug(f"Parsed subunit '{subunit}' as {su}")
        for unit, aliases in base_units_aliases.items():
            for alias in aliases:
                if su["parsed_unit"].lower() == alias.lower():
                    su["base_unit"] = unit
                    break

        su.update(UNITS.set_index("lowered").loc[su["base_unit"].lower()])
        prefix = PREFIXES.loc[su["prefix"]]

        dimension_vector += su["power"] * np.array(QUANTITY_DIMENSION_VECTORS.loc[su["physical_quantity"]])
        factor *= (su["factor"] * prefix.factor) ** su["power"]
        math_name_parts.append(repr_power(f"{prefix.math_name}{su['math_name']}", su["power"], math=True))
        units_repr_parts.append(repr_power(f"{su['prefix']}{su['base_unit']}", su["power"]))

    physical_quantity_match = (QUANTITY_DIMENSION_VECTORS == dimension_vector).all(axis=1)
    physical_quantity = (
        physical_quantity_match.loc[physical_quantity_match].index[0] if physical_quantity_match.any() else "composite"
    )

    return {
        "units": " ".join(units_repr_parts),
        "math_name": r"\ ".join(math_name_parts),
        "base_units_factor": factor,
        "base_units": repr_dim_vec(dimension_vector),
        "physical_quantity": physical_quantity,
        "dimension_vector": dimension_vector,
    }


def repr_power(thing: str, power: float, math: bool = False):
    exp_numer, exp_denom = power.as_integer_ratio()

    if exp_numer % exp_denom:
        exp_string = f"{exp_numer}/{exp_denom}" if math else f"{power}"
    else:
        exp_string = f"{int(exp_numer / exp_denom)}"
    if math:
        exp_string = f"{{{exp_string}}}"

    if power == 0:
        return ""
    if power == 1:
        return thing

    return f"{thing}^{exp_string}"


def repr_dim_vec(dim_vec: pd.Series) -> str:
    base_unit_parts = []
    for unit, power in dim_vec.items():
        part = repr_power(unit, power)
        if part:
            base_unit_parts.append(part)
    return " ".join(base_unit_parts)
