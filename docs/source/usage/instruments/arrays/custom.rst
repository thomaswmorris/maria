
Custom offsets and baselines
----------------------------

We can generate an array with pre-defined focal plane offsets with a config

.. code-block:: python

    my_first_band = Band(name="my_first_band_name", center=90e9, width=20e9, NET_RJ=5e-5)
    my_other_band = Band(name="my_other_band_name", center=150e9, width=30e9, NET_RJ=5e-5)

    array = {"bands": [my_first_band, my_other_band], 
             "primary_size": 5,
             "xi": xi,
             "eta": eta,
             "gamma": [22.5, 112.5, ..., 87.5, 157.5],
             "band_name": ["my_first_band_name", "my_first_band_name", ..., "my_other_band_name", "my_other_band_name"],
             "degrees": True}

where ``xi`` and ``eta`` are each a one-dimensional array of focal plane offsets in the azimuth and elevation directions,
``gamma`` are the polarization angles, and where each value in the ``band_name`` parameter matches the name of one of the
bands in the supplied ``bands`` parameter.

Similarly, custom baselines can be supplied as e.g.

.. code-block:: python

    array = {"bands": [f090, f150], 
             "primary_size": 5, 
             "baseline_x": baseline_x,
             "baseline_y": baseline_y,
             "baseline_z": baseline_z,
             "gamma": [22.5, 112.5, ..., 87.5, 157.5],
             "band_name": ["f090", "f090", ..., "f150", "f150"],
             "degrees": True}

where ``baseline_x``, ``baseline_y``, and ``baseline_z`` define the baseline offsets in the eastern, northern, and vertical directions.

Each detector can also have its own sensitivity, by supplying a per-detector ``NEP`` (in W√s), ``NET_RJ`` (in K_RJ√s),
``NET_CMB`` (in K_CMB√s), or ``NEFD`` (in Jy√s, for a point source). These are converted to an NEP for each detector
using its band (see :ref:`noise`).


Loading detectors from a file
-----------------------------

We can also load the detectors from a table, where each column is a per-detector parameter. Headerless tables (separated
by commas or whitespace) need the ``columns`` parameter, and the ``units`` parameter converts each column from the given
units. For a table with azimuth and elevation offsets in arcseconds and a sensitivity in mJy√s for each detector, we can do

.. code-block:: python

    array = {"file": "sensitivity.dat",
             "columns": ["xi", "eta", "NEFD"],
             "units": {"xi": "arcsec", "eta": "arcsec", "NEFD": "mJy√s"},
             "bands": [my_band],
             "primary_size": 12}
