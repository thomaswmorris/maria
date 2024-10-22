import os

import numpy as np
import pytest

import maria
from maria.instrument import Band
from maria.io import fetch
from maria.map.mappers import BinMapper

here, this_filename = os.path.split(__file__)


def test_map_sim():
    map_filename = fetch("maps/cluster.fits", refresh=True)

    f090 = Band(center=90, width=20, sensitivity=5e-5)
    f150 = Band(center=150, width=30, sensitivity=5e-5)

    array = {"field_of_view": 0.02, "bands": [f090, f150], "primary_size": 50}

    instrument = maria.get_instrument(array=array)

    map_filename = fetch("maps/cluster.fits")

    input_map = maria.map.read_fits(filename=map_filename, width=0.1, center=(150, 10))

    input_map.data *= 1e3

    plan = maria.get_plan(
        scan_pattern="daisy",
        scan_options={"radius": 0.025, "speed": 0.005},  # in degrees
        duration=60,  # in seconds
        sample_rate=50,  # in Hz
        scan_center=(150, 10),
        frame="ra_dec",
    )

    sim = maria.Simulation(
        instrument,
        plan=plan,
        site="llano_de_chajnantor",
        map=input_map,
        atmosphere="2d",
    )

    tod = sim.run()

    mapper = BinMapper(
        center=(150.01, 10.01),
        frame="ra_dec",
        width=np.radians(10.0 / 60.0),
        height=np.radians(10.0 / 60.0),
        resolution=np.radians(4.0 / 3600.0),
        degrees=False,
        tod_preprocessing={
            "remove_modes": {"n": 1},
            "filter": {"f": 0.08},
            "despline": {"knot_spacing": 10},
        },
        map_postprocessing={
            "gaussian_filter": {"sigma": 1},
            "median_filter": {"size": 1},
        },
        tods=[tod],
    )

    output_map = mapper.run()

    output_map.to_fits("/tmp/test-output.fits")
