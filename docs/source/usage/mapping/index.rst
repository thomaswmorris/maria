.. _mapping:

#######
Mapping
#######

We can map a ``TOD`` (or several of them) with a ``Mapper``. The simplest possible mapper just bins the data:

.. code-block:: python

    from maria.mappers import BinMapper

    mapper = BinMapper(center=(150, 10),
                       frame="ra_dec",
                       width=1e0,
                       height=1e0,
                       resolution=5e-3,
                       tod_preprocessing={
                            "window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
                            "remove_spline": {"knot_spacing": 5},
                            },
                       map_postprocessing={
                            "gaussian_filter": {"sigma": 1},
                            "median_filter": {"size": 1},
                            },
                       units="K_RJ",
                       tods=[tod],
                       )

    output_map = mapper.run()

where we define the preprocessing to be done on the ``TOD``. We can see the output with

.. code-block:: python

    output_map.plot()
