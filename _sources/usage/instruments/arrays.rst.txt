
######
Arrays
######

To customize an array, we specify a field of view and add some `Band`.

.. code-block:: python

    array = {"primary_size": 10 # in meters
             "field_of_view": 0.5 # in degrees
             "bands": [my_band],
             }

When passed to `Instrument`, this will generate an array such that the beams do not overlap and fill up the field of view.
Instead of a `Band`, we can also pass a string that names a pre-defined band, or a mixture of the two.

.. code-block:: python

    array = {"primary_size": 10 # in meters
             "field_of_view": 0.5 # in degrees
             "bands": [my_band, "mustang2/f093"],
             }

Constructing an `Instrument` is then done as

.. code-block:: python

    instrument = maria.get_instrument(array=array)

To construct an instrument with multiple subarrays, we can create a `dict` wherein each value is a valid array:


.. code-block:: python

    subarrays = {"array1": {"array_offset": (0.1, 0), "field_of_view": 0.05, "bands": [f150]},
                "array2": {"array_offset": (-0.1, 0), "field_of_view": 0.05, "bands": [f150]},}

    instrument = maria.get_instrument(subarrays=subarrays)
