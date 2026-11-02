.. _loading-instruments:

###################
Loading instruments
###################

We can load an instrument from our own tables of detectors, and save it to a single file to share it or load it later.


Detector tables
---------------

Each row of a detector table is one detector, and each column is a per-detector parameter (like ``xi``, ``eta``,
``gamma``, ``band_name``, or a sensitivity; see :ref:`arrays`). Columns can be separated by commas or whitespace, and lines
starting with ``#`` are ignored. A table of the azimuth and elevation offsets (in arcseconds) and the point-source
sensitivity (in mJy√s) of each detector might look like

.. code-block:: text

    # az offset ["]  el offset ["]  NEFD [mJy√s]
    -412.3   118.9    87.1
    -397.4   118.9    92.6
    ...

Headerless tables need the ``columns`` parameter, and the ``units`` parameter converts each column from the given units.
We can make an instrument from this table with

.. code-block:: python

    from maria import Band, Instrument

    my_band = Band(name="my_band", center=350e9, width=38e9, efficiency=0.5)

    array = {"name": "my_array",
             "file": "my_detectors.dat",
             "columns": ["xi", "eta", "NEFD"],
             "units": {"xi": "arcsec", "eta": "arcsec", "NEFD": "mJy√s"},
             "bands": [my_band],
             "primary_size": 12} # in meters

    my_instrument = Instrument(arrays=[array],
                               az_vel_limit=6, # in deg/s
                               acc_limit=12) # in deg/s^2

    my_instrument.plot()

Tables with a header row of parameter names don't need ``columns``. Without ``units``, offsets are in degrees, and
sensitivities are in W√s (``NEP``), K_RJ√s (``NET_RJ``), K_CMB√s (``NET_CMB``), or Jy√s (``NEFD``, assuming a Gaussian
beam). Each sensitivity is converted to an NEP using the band of its detector (see :ref:`noise`), so an array with more than
one band needs a ``band_name`` column.


Saving and loading
------------------

We can save an instrument (all of its arrays, detectors, and bands) to a FITS file with

.. code-block:: python

    my_instrument.to_fits("my_instrument.fits")

and load it back with

.. code-block:: python

    my_instrument = Instrument.from_fits("my_instrument.fits")

The file has a ``DETS`` table for each array and a ``BANDS`` table with the passband and noise properties of each band, so
it can also be read without ``maria`` (e.g. with ``astropy.io.fits``).
