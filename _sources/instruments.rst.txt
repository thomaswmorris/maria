###########
Instruments
###########

The observing instrument is instantiated as an ``Instrument``.
The simplest way to get an instrument is to grab a pre-defined one, e.g.::

    # The Atacama Cosmology Telescope
    act = maria.get_instrument('ACT')

    # The Atacama Large Millimeter Array
    alma = maria.get_instrument('ALMA')

    # MUSTANG-2
    m2 = maria.get_instrument('MUSTANG-2')


+++++++++++++++++++++
Customizing passbands
+++++++++++++++++++++

.. code-block:: python

    from maria.instrument import Band

    my_band = Band(center=150, # in GHz
                   width=30, # in GHz
                   efficiency=0.5
                   sensitivity=1e-5
                   )

Note that any sensitivity units are only implicit due to the imperfect efficiency (caused by the instrument and the atmosphere).
For a heuristic, the band implicitly assumes that we are observing the zenith at the ALMA site with a PWV of 1mm, and converts the given sensitivity to the corresponding noise equivalent power.
We can customize this by setting the sensitivity as

.. code-block:: python

    my_band.set_sensitivity(1e-5, pwv=2, elevation=45, region="mauna_kea")


++++++++++++++++++
Customizing arrays
++++++++++++++++++

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
