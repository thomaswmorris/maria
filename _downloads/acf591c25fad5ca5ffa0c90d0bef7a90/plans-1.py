import maria
from maria import Planner

input_map = maria.map.load(maria.io.fetch("maps/crab_nebula.fits")) # load an example map

planner = Planner(target=input_map,
                  site="green_bank",
                  el_bounds=(60, 90)) # make a planner

plan = planner.generate_plan(total_duration=600, # in seconds
                             scan_options={"radius": input_map.width.deg / 3}, # in degrees
                             sample_rate=50) # in Hz

plan.plot()