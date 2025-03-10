########
Overview
########

Instruments are represented by an ``Instrument`` object. When we pass a string to the ``Simulation`` (for example ``instrument='act'``) just tells it to look up the ``Instrument`` object based on that string.
We could just as easily do it ourselves as

.. code-block:: python

    # The Atacama Cosmology Telescope
    act = maria.get_instrument('ACT')

    # The Atacama Large Millimeter Array
    alma = maria.get_instrument('ALMA')

    # MUSTANG-2
    m2 = maria.get_instrument('MUSTANG-2')


We can plot the angular/spatial and frequency footprint of an instrument with e.g.

.. plot:: 
   :include-source: True

    import maria

    act = maria.get_instrument('ACT')
    act.plot()



Customizing instruments
-----------------------


The ``Instrument`` is based on a list of arrays (see :ref:`arrays`), which are in turn based on a list of bands (see :ref:`bands`). The easiest way to construct an instrument from scratch is as

.. code-block:: python

    from maria import Instrument, Band

    band1 = {"center": 150, "width": 30, "NET_RJ": 1e-5}
    band2 = "act/pa5/f150"

    array1 = {"n": 1000,
            "primary_size": 10,
            "field_of_view": 0.5,
            "bands": [band1, band2]}

    array2 = "act/pa5"

    my_instrument = Instrument(arrays=[array1, array2])
