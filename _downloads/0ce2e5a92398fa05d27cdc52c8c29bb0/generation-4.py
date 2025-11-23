from maria import Array

flower_array = {"n": 400,
            "shape": "circle",
            "packing": "sunflower",
            "field_of_view": 0.5, # in degrees
            "primary_size": 10, # in meters
            "bands": ["act/pa5/f090", "act/pa5/f150"]}

Array.from_config(flower_array).plot()