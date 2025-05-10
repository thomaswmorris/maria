from __future__ import annotations

import os

import numpy as np
import pytest

import maria
from maria import Simulation, all_instruments
from maria.mappers import BinMapper
from maria.utils import read_yaml

here, this_filename = os.path.split(__file__)


def test_polarized_map_sim():
    plan = maria.Plan(
        start_time="2024-08-06T09:00:00",
        scan_pattern="daisy",
        scan_options={"radius": 0.5, "speed": 0.1},  # in degrees
        duration=5,  # in seconds
        sample_rate=20,  # in Hz
        scan_center=(31, 62),
        frame="az_el",
    )

    sim = Simulation(
        instrument="test/1deg",
        site="llano_de_chajnantor",
        plan=plan,
        atmosphere="2d",
        cmb="generate",
        map="maps/einstein.h5",
    )

    tod = sim.run()

    for field in ["atmosphere", "cmb"]:
        if np.isnan(tod.data[field]).any():
            raise ValueError(f"There are NaNs in the '{field}' field.")

    tod = tod.to("K_RJ")

    mapper = BinMapper(
        center=(0, -23),
        stokes="IQUV",
        frame="ra_dec",
        width=1,
        height=1,
        resolution=1 / 256,
        tod_preprocessing={
            "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
            "remove_modes": {"modes_to_remove": [0]},
        },
        map_postprocessing={
            "gaussian_filter": {"sigma": 1},
            "median_filter": {"size": 1},
        },
        units="mK_RJ",
    )

    mapper.add_tods(tod)
    mapper.run()
