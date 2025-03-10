from maria import Array

my_array = {"n": 217,
            "field_of_view": 0.5, # in degrees
            "primary_size": 10, # in meters
            "bands": ["act/pa5/f090", "act/pa5/f150"]}

Array.from_config(my_array).plot()