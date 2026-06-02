from maria import Instrument, Band

band1 = {"center": 150, "width": 30, "NET_RJ": 1e-5}
band2 = "act/pa5/f150"

array1 = {"n": 1000,
        "primary_size": 10, # in meters
        "field_of_view": 0.5, # in degrees
        "bands": [band1, band2]}

array2 = "act/pa5"

my_instrument = Instrument(arrays=[array1, array2])