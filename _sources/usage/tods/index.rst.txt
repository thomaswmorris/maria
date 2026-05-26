####
TODs
####

The result of a simulation is a ``TOD``, which encapsulates the generated time-ordered data (and all metadata, like the pointing coordinates and detectors).

We can see what our data looks like with

.. code-block:: python

    tod.plot()

A ``TOD`` can be sliced like any array; we can get the first ten thousand samples for every other detector as

.. code-block:: python

    subtod = tod[::2, :10000]

==========
Components
==========

The total signal in each detector can be accessed as

.. code-block:: python

    tod.signal # returns an array

which is the sum of all of the simulated fields (e.g. noise, atmosphere, CMB) separately, contributing to the incident power. For convenience, we can also access the individual fields as

.. code-block:: python

    tod.get_field("atmosphere") # returns an array

We can see all the available fields with ``tod.fields``.

=====
Units
=====

TODs are by default in units of picowatts, but we can convert to any unit that is a combination of an SI prefix and a base unit (one of `K_RJ`, `K_CMB`, or `W`).

.. code-block:: python

    tod_in_rj_units = tod.to(units="mK_RJ")
    tod_in_cmb_units = tod.to(units="uK_CMB")

=============
Load and save
=============

We can output TODs to disk,
.. code-block:: python

    tod.to_fits('filename.fits')

or if you rather work with hdf files,

.. code-block:: python

    tod.to_hdf('filename.hdf5')

you can load fits files back with

.. code-block:: python

    tod = TOD.from_fits('filename.fits', format='MUSTANG-2')

note, that you can also load in real MUSTANG-2 data with the same command. 
