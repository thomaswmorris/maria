import maria
from maria import Planner
from maria.mappers import BinMapper

import numpy as np
import matplotlib.pyplot as plt

# --- input map (Crab Nebula / M1) ---
input_map = maria.map.get("maps/m1.h5")

# --- scanning strategy: 900 s, constrained to el > 60 deg ---
planner = Planner(target=input_map, site="green_bank", constraints={"el": (60, 90)})
plans = planner.generate_plans(total_duration=900, sample_rate=100)

# --- instrument ---
instrument = maria.get_instrument("MUSTANG-2")

# --- simulation ---
sim = maria.Simulation(
    instrument,
    plans=plans,
    site="green_bank",
    map=input_map,
    atmosphere="2d",
)

tods = sim.run()

# --- map recovery ---
mapper = BinMapper(
    tods=tods,
    units="uK_RJ",
    resolution=0.75*input_map.resolution,
    tod_preprocessing={
        "remove_modes": {"modes_to_remove": 2},
        "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
    },
    map_postprocessing = {}
)
output_map = mapper.run()

# --- transfer function ---
tf = output_map.transfer_function()
tf.plot(x_unit="arcsec", filename="transfer_function.png")

raise SystemExit

tfs = []
for _ in range(10):
    tods = sim.run()

    # --- map recovery ---
    mapper = BinMapper(
        tods=tods,
        units="uK_RJ",
        resolution=0.75*input_map.resolution,
        tod_preprocessing={
            "remove_modes": {"modes_to_remove": 1},
            "remove_spline": {"knot_spacing": 30, "remove_el_gradient": True},
        },
        map_postprocessing = {}
    )
    output_map = mapper.run()

    # --- transfer function ---
    tf = output_map.transfer_function()
    tfs.append(tf)

tfs_avg = np.mean([tf.T for tf in tfs], axis=0)
tfs_std = np.std([tf.T for tf in tfs], axis=0)

plt.plot(tfs[0].u, tfs_avg, label="average")
plt.show(); plt.close()