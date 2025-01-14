from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import maria
import numpy as np
from maria.io import fetch

from maria.instrument import Band
from maria.mappers import BinMapper

plt.close("all")


@pytest.mark.parametrize(
    "map_name",
    ["maps/cluster.fits", "maps/big_cluster.fits", "maps/galaxy.fits"],
)  # noqa
def test_fetch_fits_map(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(
        filename=map_filename,
        nu=90,
        resolution=1 / 1024,
        center=(150, 10),
        units="Jy/pixel",
    )

    m.plot()


@pytest.mark.parametrize(
    "map_name",
    ["maps/cluster.fits", "maps/big_cluster.fits", "maps/galaxy.fits"],
)  # noqa
def test_map_units_conversion(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(
        filename=map_filename,
        nu=90,
        resolution=1 / 1024,
        center=(150, 10),
        units="Jy/pixel",
    )

    assert np.allclose(m.to("K_RJ").to("Jy/pixel").data, m.data).compute()


@pytest.mark.parametrize("map_name", ["maps/sun.h5"])  # noqa
def test_fetch_hdf_map(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(filename=map_filename)
    m.plot()


def test_trivial_recover_original_map():

    f090 = Band(center=90, width=30)
    f150 = Band(center=150, width=30)
    f220 = Band(center=220, width=30)

    array = {
        "field_of_view": 0.08333,
        "primary_size": 100,
        "n": 313,
        "bands": [f090, f150, f220],
    }

    instrument = maria.get_instrument(array=array)

    map_filename = fetch("maps/big_cluster.fits", refresh=True)
    input_map = maria.map.read_fits(
        filename=map_filename, nu=150, width=0.1, center=(150, 10)
    ).downsample((1, 1, 1, 100, 100))

    plan = maria.Plan(
        scan_pattern="daisy",
        scan_options={"radius": 0.5, "speed": 1},  # in degrees
        duration=60,  # in seconds
        sample_rate=50,  # in Hz
        scan_center=np.degrees(input_map.center),
        jitter=0,
        frame="ra_dec",
    )

    sim = maria.Simulation(
        instrument,
        plan=plan,
        map=input_map,
        site="cerro_toco",
        noise=False,
    )

    tod = sim.run()

    mapper = BinMapper(
        center=input_map.center,
        frame=input_map.frame,
        width=input_map.width,
        resolution=input_map.resolution,
        degrees=False,
    )

    mapper.add_tods(tod)
    output_map = mapper.run()

    weight = output_map.weight / output_map.weight.sum()
    residuals = (output_map.data - input_map.data).compute()
    relative_residual = (
        np.nanstd(weight * residuals).compute() / np.nanstd(input_map.data).compute()
    )

    assert relative_residual < 1e-3


def test_time_ordered_map_sim():

    time_evolving_sun_path = fetch("maps/sun.h5")
    input_map = maria.map.load(
        filename=time_evolving_sun_path, t=1.7e9 + np.linspace(0, 180, 16)
    )
    plan = maria.Plan(
        start_time=1.7e9,
        duration=180,
        scan_center=np.degrees(input_map.center),
        scan_options={"radius": 0.25},
    )
    sim = maria.Simulation(plan=plan, map=input_map)
    tod = sim.run()
    tod.to("K_RJ").plot()
