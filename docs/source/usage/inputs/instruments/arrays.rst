.. _arrays:
######
Arrays
######

An ``Array`` is a set of detectors, either at the same point in space distributed across some field of view (e.g. a bolometric camera) or distributed in space looking at the same point on the sky (e.g. an interferometer). 
To generate a custom ``Array`` we specify a configuration as a ``dict``, for example as

.. plot:: 
   :include-source: True

       from maria import Array

       my_array = {"n": 217,
                   "field_of_view": 0.5, # in degrees
                   "primary_size": 10, # in meters
                   "bands": ["act/pa5/f090", "act/pa5/f150"]}

       Array.from_config(my_array).plot()

where the ``bands`` parameter is a list of either ``Band`` object or a string corresponding to a pre-defined band.

.. note::

    The items in the list supplied as the ``bands`` parameter can be a ``Band`` object, a ``dict``, or a ``str``. The following are all valid bands:

    .. code-block:: python

        band_1 = Band(center=150, width=30, NET_RJ=1e-5)

        band_2 = {"center": 90, "width": 30, "NEP": 1e-15}

        band_3 = "act/pa5/f150"

.. hint:: To see all available pre-defined bands, run ``print(maria.all_bands)``. For documentation on how to define a custom band, refer to the :ref:`bands` section.

We can then generate an ``Instrument`` as

.. code-block:: python
    
    from maria import Instrument

    instrument = Instrument(arrays=[my_array])

which generates an array of 1000 multichroic detectors (a detector per band per array position).

.. note::

    The items in the ``arrays`` list can be an ``Array`` object, a ``dict``, or a ``str``. The following are all valid arrays that can be in an ``arrays`` list:

    .. code-block:: python

        array_1 =  {"n": 217,
                   "field_of_view": 0.5, # in degrees
                   "primary_size": 10, # in meters
                   "bands": ["act/pa5/f090", "act/pa5/f150"]}
                   
        array_2 = Array.from_config(array_1)

        array_3 = "so/sat-wafer"

We can also leave the number of detectors ``n`` implicit and define an array with a given field of view as

.. code-block:: python

    array = {"primary_size": 10,
             "field_of_view": 0.5,
             "bands": [band_1, band_2]}

Here, ``maria`` will infer the number of detectors by packing the field of view with beams of a given resolution (determined by the ``primary_size`` and ``bands`` parameters). 
By default, the spacing between the beams will be 1.5 times the FWHM, but we can adjust the ratio with the ``beam_spacing`` parameter.

.. _multiple-arrays:
Multiple arrays
---------------

Many instruments are made up of individual subarrays that set next to each other on the focal plane. We can create an instrument made of identical subarrays with

.. plot:: 
   :include-source: True

    from maria import Instrument

    array = {"primary_size": 5,
             "field_of_view": 1.0,
             "bands": ["act/pa5/f090", "act/pa5/f150"]}

    subarray_left = {"focal_plane_offset": (-1, 0), **array}
    subarray_right = {"focal_plane_offset": (1, 0), **array}

    instrument = Instrument(arrays=[subarray_left, subarray_right])

    instrument.plot()


Polarized arrays
----------------

We can define an array with polarized detectors as

.. code-block:: python

    array = {"n": 1000,
             "field_of_view": 0.5,
             "primary_size": 10,
             "polarized": True,
             "bands": [my_band]}

which will generate two orthogonally polarized detectors per band per array position.


Array shapes and packings
-------------------------

By default, the generated array is hexagonal with a triangular packing. We can change both of these with the ``shape`` and ``packing`` parameters. Consider

.. plot:: 
   :include-source: True

       from maria import Array

       octagonal_array = {"n": 1005,
                          "shape": "octagon",
                          "packing": "square",
                          "field_of_view": 0.5, # in degrees
                          "primary_size": 25, # in meters
                          "bands": ["act/pa5/f090", "act/pa5/f150"]}

       Array.from_config(octagonal_array).plot()

or 

.. plot:: 
   :include-source: True

       from maria import Array

       flower_array = {"n": 400,
                   "shape": "circle",
                   "packing": "sunflower",
                   "field_of_view": 0.5, # in degrees
                   "primary_size": 10, # in meters
                   "bands": ["act/pa5/f090", "act/pa5/f150"]}

       Array.from_config(flower_array).plot()

To generate a long, thin array, we parametrize the array in terms of the numbers of rows and columns:

.. plot:: 
   :include-source: True

       from maria import Array

       stripe_array = {"n_col": 5,
                       "n_row": 25,
                       "shape": "square",
                       "packing": "triangular",
                       "field_of_view": 0.5,
                       "primary_size": 15,
                       "bands": ["act/pa5/f090", "act/pa5/f150"]}

       Array.from_config(stripe_array).plot()
