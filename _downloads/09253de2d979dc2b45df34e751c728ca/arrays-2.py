from maria import Instrument

array = {"primary_size": 5,
         "field_of_view": 1.0,
         "bands": ["act/pa5/f090", "act/pa5/f150"]}

subarray_left = {"focal_plane_offset": (-1, 0), **array}
subarray_right = {"focal_plane_offset": (1, 0), **array}

instrument = Instrument(arrays=[subarray_left, subarray_right])

instrument.plot()