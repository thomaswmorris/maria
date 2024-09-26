####
Maps
####

A map is represented by a ``Map`` object. We can load some pre-defined ones with

.. code-block:: python

    from maria.io import fetch

    map_filename = fetch("maps/big_cluster.fits")

    input_map = maria.map.read_fits(filename=map_filename,
                                    index=1,
                                    resolution=1/1024,
                                    center=(150, 10),
                                    units="Jy/pixel")

    input_map.to(units="K_RJ").plot()
