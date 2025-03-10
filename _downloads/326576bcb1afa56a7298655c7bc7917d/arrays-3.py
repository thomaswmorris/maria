from maria import Array

octagonal_array = {"n": 1005,
                   "shape": "octagon",
                   "packing": "square",
                   "field_of_view": 0.5, # in degrees
                   "primary_size": 25, # in meters
                   "bands": ["act/pa5/f090", "act/pa5/f150"]}

Array.from_config(octagonal_array).plot()