from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import maria
from maria.instrument import Band
from maria.io import fetch
from maria.mappers import BinMapper

plt.close("all")


@pytest.mark.parametrize(
    "map_name",
    ["maps/cluster.fits", "maps/big_cluster.h5", "maps/galaxy.fits"],
)  # noqa
def test_fetch_fits_map(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(
        filename=map_filename,
        nu=90e9,
        resolution=1 / 1024,
        center=(150, 10),
        units="Jy/pixel",
    )

    m.plot()


@pytest.mark.parametrize(
    "map_name",
    ["maps/cluster.fits", "maps/big_cluster.h5", "maps/galaxy.fits"],
)  # noqa
def test_map_units_conversion(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(
        filename=map_filename,
        nu=90e9,
        resolution=1 / 1024,
        center=(150, 10),
        units="Jy/pixel",
    )

    assert np.allclose(m.to("K_RJ").to("Jy/pixel").data, m.data).compute()


@pytest.mark.parametrize("map_name", ["maps/sun.h5"])  # noqa
def test_fetch_hdf_map(map_name):
    map_filename = fetch(map_name)
    m = maria.map.load(filename=map_filename, nu=100e9)
    m.plot()


def test_trivial_recover_original_map():
    f090 = Band(center=90e9, width=30e9)
    f150 = Band(center=150e9, width=30e9)
    f220 = Band(center=220e9, width=30e9)

    array = {
        "field_of_view": 0.08333,
        "primary_size": 100,
        "n": 313,
        "bands": [f090, f150, f220],
    }

    instrument = maria.get_instrument(array=array)

    map_filename = fetch("maps/big_cluster.h5", refresh=True)
    input_map = maria.map.load(filename=map_filename, nu=150e9, width=0.1, center=(150, 10)).downsample(n_x=100, n_y=100)

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
        resolution=input_map.x_res,
        degrees=False,
    )

    mapper.add_tods(tod)
    output_map = mapper.run()

    weight = output_map.weight / output_map.weight.sum()
    residuals = (output_map.data - input_map.data).compute()
    relative_residual = np.nanstd(weight * residuals).compute() / np.nanstd(input_map.data).compute()

    assert relative_residual < 1e-3


def test_time_ordered_map_sim():
    time_evolving_sun_path = fetch("maps/sun.h5")
    input_map = maria.map.load(filename=time_evolving_sun_path, nu=100e9, t=1.7e9 + np.linspace(0, 180, 16))
    plan = maria.Plan(
        start_time=1.7e9,
        duration=180,
        scan_center=np.degrees(input_map.center),
        scan_options={"radius": 0.25},
    )
    sim = maria.Simulation(plan=plan, map=input_map)
    tod = sim.run()
    tod.to("K_RJ").plot()


@pytest.mark.parametrize("filename", ["big_cluster.h5", "cluster.fits"])
def test_maps_io(filename):
    map_filename = fetch(f"maps/{filename}")
    m = maria.map.load(filename=map_filename, width=1e0, units="K_RJ")

    m.to("cK_RJ").to_fits("/tmp/test.fits")

    new_m = maria.map.load("/tmp/test.fits").to("MK_RJ")
    new_m.to_hdf("/tmp/test.h5")

    new_new_m = maria.map.load("/tmp/test.h5").to("K_RJ")  # noqa

    assert np.allclose(new_new_m.data, m.data)
    assert new_new_m.resolution == m.resolution
