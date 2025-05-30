from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import maria
from maria.instrument import Band
from maria.io import fetch
from maria.map import all_maps
from maria.mappers import BinMapper

plt.close("all")


@pytest.mark.parametrize(
    "map_path",
    all_maps,
)  # noqa
def test_maps(map_path):
    m = maria.map.load(fetch(map_path))
    if "nu" not in m.dims:
        m = m.unsqueeze("nu", 150e9)
    assert np.allclose(m.to("K_RJ").to("Jy/pixel").to(m.units).data, m.data).compute()

    m.to("cK_RJ").to_hdf("/tmp/test.h5")
    new_m = maria.map.load("/tmp/test.h5").to(m.units)  # noqa

    assert np.allclose(new_m.data, m.data)
    assert np.allclose(new_m.resolution, m.resolution)

    m.plot()


def test_trivial_recover_original_map():
    f090 = Band(center=90e9, width=30e9)
    f150 = Band(center=150e9, width=30e9)
    f220 = Band(center=220e9, width=30e9)

    map_filename = fetch("maps/cluster1.fits")
    input_map = maria.map.load(filename=map_filename, nu=150e9).to("K_RJ")[..., ::8, ::8]

    input_map.data -= input_map.data.mean().compute()
    map_width = np.degrees(input_map.width)

    array = {
        "field_of_view": map_width / 2,
        "primary_size": 1000,
        "n": 300,
        "bands": [f090, f150, f220],
    }

    instrument = maria.get_instrument(array=array)

    plan = maria.Plan(
        scan_pattern="daisy",
        scan_options={"radius": map_width / 4, "speed": map_width / 10},  # in degrees
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

    m0 = input_map.data[0, 0, 0].compute()
    m1 = output_map.data[0, :].compute()
    w = output_map.weight[-1, 0].compute()

    relsqres = np.sqrt(np.nansum(w * (m1 - m0) ** 2, axis=(-1, -2)) / np.nansum(w))

    assert all(relsqres < 1e-3)


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
