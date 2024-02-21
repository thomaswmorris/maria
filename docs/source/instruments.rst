===========
Instruments
===========

The observing instrument is instantiated as an ``Instrument``.
The simplest way to get an instrument is to grab a pre-defined one, e.g.::

    # The Atacama Cosmology Telescope
    act = maria.get_instrument('ACT')

    # The Atacama Large Millimeter Array
    alma = maria.get_instrument('ALMA')

    # MUSTANG-2
    m2 = maria.get_instrument('MUSTANG-2')


+++++++++++++++++++++++
Customizing Instruments
+++++++++++++++++++++++

One way to customize instruments is to load a pre-defined instrument and pass extra parameters.
For example, we can give ACT twice the resolution with

.. code-block:: python

    act = maria.get_instrument('ACT', primary_size=12)


Custom arrays of detectors are a bit more complicated. For example:

.. code-block:: python

    f090 = {"center": 90, "width": 30}
    f150 = {"center": 150, "width": 30}

    dets = {"n": 500,
            "field_of_view": 2
            "array_shape": "hex",
            "bands": [f090, f150]}

    my_custom_array = maria.get_instrument(dets=dets)

Actually, there are several valid ways to define an array of detectors.
These are all valid:

.. code-block:: python

    f090 = {"center": 90, "width": 30}
    f150 = {"center": 150, "width": 30}

    dets = {"n": 500,
            "field_of_view": 2
            "array_shape": "hex",
            "bands": [f090, f150]}

    my_custom_array = maria.get_instrument(dets=dets)
