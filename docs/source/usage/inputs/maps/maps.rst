
.. _maps:

####
Maps
####

++++++++
Overview
++++++++

A map is represented by a ``Map`` object, that represents a two-dimensional slice (or many slices) of radiometric data. 

We can download a pre-defined ``maria`` maps with

.. plot:: 
   :include-source: True

    import maria

    m = maria.map.get("maps/M1.h5")

    print(m)
    m.plot(slices="all")

.. hint:: To see all available pre-defined maps, run ``print(maria.all_maps)``. For documentation on customizing these inputs, see the :ref:`instruments`, :ref:`sites`, and :ref:`plans` sections.


+++++++++++++++++++
Loading from a file
+++++++++++++++++++

If you have a local map file (either as an ``.h5`` file or a ``.fits`` file), you can load it into ``maria`` as

.. code-block:: python

    m = maria.map.load("my_fits_map.fits")
    m = maria.map.load("my_hdf_map.h5")

When constructing the map, ``maria`` will try to infer map metadata from the file. 

.. warning:: ``maria`` can handle most FITS (Flexible Image Transport System) files, but the "flexible" part means that there are too many possible conventions for ``maria`` to be able to parse reliably. If ``maria`` cannot parse a FITS file header, you can specify the map metadata manually as demonstrated below.

Passing additional arguments to ``maria.map.load`` will override any metadata. For example, running 

.. code-block:: python

    m = maria.map.load("my_fits_map.fits", width=1, units="uK_RJ", nu=150e9)

will scale the size of the map to be one degree wide, and interpret the map values as being in units of microKelvin Rayleigh-Jeans at 150GHz.

.. warning:: When overriding metadata, any passed dimensions must match the shape of the map; passing a single frequency to a three-dimensional map with dimensions ``(nu, y, x) = (10, 256, 256)`` will throw an error.

+++++
Units
+++++

A ``Map`` in ``maria`` can be represented in several different units.

+-------------------------------------+--------------------------------------+
| Quantity                            | Units                                |
+=====================================+======================================+
| Rayleigh-Jeans temperature          | ``K_RJ``                             |
+-------------------------------------+--------------------------------------+
| Brightness temperature              | ``K_b``                              |
+-------------------------------------+--------------------------------------+
| Spectral flux density per unit area | ``Jy/pixel``, ``Jy/beam``, ``Jy/sr`` |
+-------------------------------------+--------------------------------------+
| Compton y                           | ``y``                                |
+-------------------------------------+--------------------------------------+

``maria`` will try to parse a given units string (e.g. aliases like ``Jy/pix`` or ``Jy pix**-1`` for ``Jy/pixel``) as well as any valid SI prefix (like ``mK_RJ``).

A map can be converted from one set of units to another as 

.. code-block:: python

    map_in_K_RJ = map_in_some_other_units.to("K_RJ")

Note that while a map can have units of e.g. ``Jy/pixel`` without any frequency information, it cannot be converted to another quantity without specifying a frequency.
We can add a frequency dimension to a map as

.. code-block:: python

    map_with_frequency_dim = map_without_frequency_dim.unsqueeze("nu", 150e9)



+++++++++++++++++++
Manual construction
+++++++++++++++++++

We can manually create a map from a two-dimensional array of data as

.. code-block:: python

    from maria.map import ProjectionMap

    m = ProjectionMap(data=data, # an array with shape (n_y, n_x)
                      weight=weight, # optional, must be the same shape as 'data'
                      units=units, # required
                      center=(ra, dec), # or (glon, glat)
                      frame="ra/dec", # or "galactic"
                      resolution=1e-3, # spacing between pixels, in degrees
    )


To create a map for data with more dimensions, we 

.. code-block:: python

    from maria.map import ProjectionMap

    m = ProjectionMap(data=data, # an array with shape e.g. (stokes, freq, y, x) = (4, 3, 256, 256)
                      weight=weight, # optional, must be the same shape as 'data'
                      units=units, # required
                      center=(ra, dec), # or (glon, glat)
                      frame="ra/dec", # or "galactic"
                      resolution=1e-3, # spacing between pixels, in degrees
                      stokes="IQUV",
                      nu=[nu1, nu2, nu3],
    )


++++++
Slices
++++++

+--------------------+--------------------+--------------------------+---------------------------------------------------------------+
| Dimension          | Description        | Default units            | Valid values                                                  |
+====================+====================+==========================+===============================================================+
| ``stokes``         | Stokes parameter   | None                     | A combination of ``["I", "Q", "U", "V"]`` or ``[0, 1, 2, 3]`` |
+--------------------+--------------------+--------------------------+---------------------------------------------------------------+
| ``nu``             | Frequency          | ``Hz``                   | Any positive number                                           | 
+--------------------+--------------------+--------------------------+---------------------------------------------------------------+
| ``z``              | Redshift           | None                     | Any positive number                                           |
+--------------------+--------------------+--------------------------+---------------------------------------------------------------+
| ``v``              | Radial velocity    | ``m/s``                  | Any number                                                    |
+--------------------+--------------------+--------------------------+---------------------------------------------------------------+
| ``t``              | Redshift           | ``seconds`` (UNIX epoch) | Any number                                                    |
+--------------------+--------------------+--------------------------+---------------------------------------------------------------+


++++++++++++
HEALPix maps
++++++++++++

Defining a HEALPix map is much the same as with a two-dimensional map:

.. code-block:: python

    from maria.map import HEALPixMap

    m = HEALPixMap(data=data, # an array with shape e.g. (stokes, freq, pixels) = (4, 3, 786432)
                   weight=weight, # optional, must be the same shape as 'data'
                   units=units, # required
                   frame="ra/dec", # or "galactic"
                   resolution=1e-3, # spacing between pixels, in degrees
                   stokes="IQUV",
                   nu=[nu1, nu2, nu3],
    )
