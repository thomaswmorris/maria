===========
Pointings
===========

The observing pointing is instantiated as an ``Pointing``.
The simplest way to get an pointing is to grab a pre-defined one, e.g.::

    # The Atacama Cosmology Telescope
    act = maria.get_pointing('ACT')

    # The Atacama Large Millimeter Array
    alma = maria.get_pointing('ALMA')

    # MUSTANG-2
    m2 = maria.get_pointing('MUSTANG-2')


+++++++++++++++++++++++
Customizing Pointings
+++++++++++++++++++++++

One way to customize pointings is to load a pre-defined pointing and pass extra parameters.
For example, we can give ACT twice the resolution with

.. code-block:: python

    act = maria.get_pointing('ACT', primary_size=12)


Custom arrays of detectors are a bit more complicated. For example:

.. code-block:: python

    f090 = {"center": 90, "width": 30}
    f150 = {"center": 150, "width": 30}

    dets = {"n": 500,
            "field_of_view": 2
            "array_shape": "hex",
            "bands": [f090, f150]}

    my_custom_array = maria.get_pointing(dets=dets)

Actually, there are several valid ways to define an array of detectors.
These are all valid:

.. code-block:: python


    dets = {
        "subarray-1": {
            "n": 500,
            "field_of_view": 2,
            "array_shape": "hex",
            "bands": [{"center": 30, "width": 5}, {"center": 40, "width": 5}],
        },
        "subarray-2": {
            "n": 500,
            "field_of_view": 2,
            "array_shape": "hex",
            "bands": [{"center": 90, "width": 5}, {"center": 150, "width": 5}],
        },
    }

    dets = {
        "n": 500,
        "field_of_view": 2,
        "array_shape": "hex",
        "bands": ["alma/f043", "alma/f078"],
    }

    dets = {"file": "path_to_some_dets_file.csv"}
