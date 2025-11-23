
Custom offsets and baselines
----------------------------

We can generate an array with pre-defined focal plane offsets with a config

.. code-block:: python

    my_first_band = Band(name="my_first_band_name", center=90e9, width=20e9, NET_RJ=5e-5)
    my_other_band = Band(name="my_other_band_name", center=150e9, width=30e9, NET_RJ=5e-5)

    array = {"bands": [my_first_band, my_other_band], 
             "primary_size": 5,
             "sky_x": sky_x, 
             "sky_y": sky_y,
             "pol_angle": [22.5, 112.5, ..., 87.5, 157.5],
             "band_name": ["my_first_band", "my_first_band", ..., "my_other_band", "my_other_band"],
             "degrees": True}

where ``sky_x`` and ``sky_y`` are each a one-dimensional array of focal plane offsets in the x and y directions, and where 
each value in the ``band_name`` parameter matches the name of one of the bands in the supplied ``bands`` parameter.

Similarly, custom baselines can be supplied as

.. code-block:: python

    array = {"bands": [f090, f150], 
             "primary_size": 5, 
             "baseline_x": baseline_x,
             "baseline_y": baseline_y,
             "baseline_z": baseline_z,
             "pol_angle": [22.5, 112.5, ..., 87.5, 157.5],
             "band_name": ["f090", "f090", ..., "f150", "f150"],
             "degrees": True}

where ``baseline_x``, ``baseline_y``, and ``baseline_z`` define the baseline offsets in the eastern, northern, and vertical directions. 
