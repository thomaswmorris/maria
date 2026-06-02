.. _mapping:

#######
Mapping
#######

We can map a ``TOD`` (or several of them) with a ``Mapper``. The simplest possible mapper just bins the data:

.. code-block:: python

    from maria.mappers import MaximumLikelihoodMapper

    ml_mapper = MaximumLikelihoodMapper(tods=[tod1, tod2, ...],
                                        tod_preprocessing={
                                            "remove_spline": {"knot_spacing": 600, "remove_el_gradient_order": 3},
                                        },
                                        init="random",
                                        bilinear=False,
                                        prior=False)

    ml_mapper.fit(epochs=4,
                  steps_per_epoch=25,
                  plot=True)

    output_map = ml_mapper.map

We can see the output map with

.. code-block:: python

    output_map.plot()


#############
Customization
#############

By default, the mapper infers the dimensions and resolution of the map based on the patch of sky covered by the supplied ``TOD``.
But we can customize the dimensions by optionally passing ``center``, ``width``, ``height``, and ``resolution`` parameters.
By default the mapper converts the data to units of ``K_RJ`` (Kelvin Rayleigh-Jeans), 
but we can also map the data in units of ``Jy/pixel`` (Janskies per pixel) or ``K_CMB`` (CMB anisotropy temperature)

.. code-block:: python

    mapper = BinMapper(center=(150, 10),
                       frame="ra/dec",
                       width=1e0,
                       height=1e0,
                       resolution=5e-3,
                       units="K_RJ",
                       tods=[tod1, tod2, ...],
                       )

#######################
Pre- and postprocessing
#######################

We specify the preprocessing we want done to each ``TOD`` and the postprocessing done to the output map also

.. code-block:: python

    mapper = BinMapper(tod_preprocessing={
                            "remove_spline": {"knot_spacing": 10},
                            },
                       map_postprocessing={
                            "gaussian_filter": {"sigma": 1},
                            "median_filter": {"size": 1},
                            },
                       tods=[tod1, tod2, ...],
                       )



###################
Time evolution maps
###################

To map a time-evolving source, we can pass a ``timestep`` parameter as

.. code-block:: python

    mapper = BinMapper(tods=[tod1, tod2, ...],
                       timestep=60,
                       )

to make a map of the source every minute.
