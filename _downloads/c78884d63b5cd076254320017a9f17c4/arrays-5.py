from maria import Array

stripe_array = {"n_col": 5,
                "n_row": 25,
                "shape": "square",
                "packing": "triangular",
                "field_of_view": 0.5,
                "primary_size": 15,
                "bands": ["act/pa5/f090", "act/pa5/f150"]}

Array.from_config(stripe_array).plot()