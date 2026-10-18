

########
Overview
########

The observing site is represented by a ``Site``. The easiest way to instantiate a ``Site`` is to load a preset one as

.. plot:: 
   :include-source: True

    import maria

    cerro_toco = maria.get_site("cerro_toco")
    cerro_toco.plot()

.. hint:: To see all available pre-defined sites, run ``print(maria.all_sites)``.

We can design a ``Site`` from scratch as 

.. code-block:: python

    from maria import Site

    my_site = Site(latitude=latitude, 
                   longitude=longitude, 
                   altitude=altitude,
                   region=region)

where ``latitude`` and ``longitude`` are in degrees, ``altitude`` is in meters, and ``region`` determines the weather data and the atmospheric emission/transmission spectrum model ``maria`` uses for simulations based on that ``Site``.

.. hint:: To see all available pre-defined regions, run ``print(maria.all_regions)``.
