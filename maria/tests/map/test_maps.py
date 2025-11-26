from __future__ import annotations

import maria
import matplotlib.pyplot as plt
import numpy as np
import pytest
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

    m.to("cK_RJ").to_hdf("/tmp/test_write_map.h5")
    new_m_hdf = maria.map.load("/tmp/test_write_map.h5").to(m.units)  # noqa

    assert np.allclose(new_m_hdf.data, m.data)
    assert np.allclose(new_m_hdf.resolution.arcsec, m.resolution.arcsec)

    if "fits" in map_path:
        m.to("cK_RJ").to_fits("/tmp/test_write_map.fits")
        new_m_fits = maria.map.load("/tmp/test_write_map.fits").to(m.units)  # noqa

        assert np.allclose(new_m_fits.data, m.data)
        assert np.allclose(new_m_fits.resolution.arcsec, m.resolution.arcsec)

    m.plot()

    plt.close("all")


def test_map_operations():
    map_filename = fetch("maps/cluster1.fits")

    m1 = maria.map.load(filename=map_filename, nu=90e9)
    m2 = maria.map.load(filename=map_filename, nu=150e9)
    m3 = maria.map.load(filename=map_filename, nu=220e9)

    m4 = m1.extend([m2, m3], dim="nu").unsqueeze("stokes")
    m5, m6 = m4.copy(), m4.copy()
    m5.stokes = "Q"
    m6.stokes = "U"

    m4.extend([m5, m6], dim="stokes")


def test_trivial_recover_original_map():
    f090 = Band(center=90e9, width=30e9)
    f150 = Band(center=150e9, width=40e9)
    f220 = Band(center=220e9, width=50e9)

    map_filename = fetch("maps/cluster1.fits")
    input_map = maria.map.load(filename=map_filename, nu=150e9).to("K_RJ")[..., ::8, ::8]

    input_map.data -= input_map.data.mean().compute()
    map_width = input_map.width.degrees

    array = {
        "field_of_view": map_width / 2,
        "primary_size": 1000,
        "n": 300,
        "bands": [f090, f150, f220],
    }

    instrument = maria.get_instrument(array=array)

    planner = maria.Planner(target=input_map, site="cerro_toco", constraints={"el": (40, 90)})

    plans = planner.generate_plans(
        total_duration=60, sample_rate=50, scan_pattern="daisy", scan_options={"radius": input_map.width.deg / 3}
    )

    sim = maria.Simulation(
        instrument,
        plans=plans,
        map=input_map,
        site="cerro_toco",
        noise=False,
    )

    tods = sim.run()

    mapper = BinMapper(
        center=input_map.center,
        frame=input_map.frame,
        width=input_map.width,
        resolution=input_map.x_res,
        degrees=False,
        tods=tods,
    )

    mapper.add_tods(tods)
    output_map = mapper.run()

    m0 = input_map.data[0, 0, 0].compute()
    m1 = output_map.data[0, :].compute()
    w = output_map.weight[-1, 0].compute()

    relsqres = np.sqrt(np.nansum(w * (m1 - m0) ** 2, axis=(-1, -2)) / np.nansum(w))

    assert all(relsqres < 1e-3)


def test_time_ordered_map_sim():
    time_evolving_sun_path = fetch("maps/sun.h5")
    input_map = maria.map.load(filename=time_evolving_sun_path, nu=100e9, t=1.7e9 + np.linspace(0, 180, 16))
    plan = maria.Plan.generate(
        start_time=1.7e9,
        duration=180,
        scan_center=(input_map.center),
        scan_options={"radius": 0.25},
    )
    sim = maria.Simulation(instrument="test/1deg", site="cerro_toco", plans=plan, map=input_map)
    tods = sim.run()
    tods[0].to("K_RJ").plot()

    plt.close("all")
