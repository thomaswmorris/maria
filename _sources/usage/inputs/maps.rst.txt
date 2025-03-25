
.. _maps:
####
Maps
####

A map is represented by a ``Map`` object. We can load some pre-defined ones with

.. plot:: 
   :include-source: True

    import maria

    map_filename = maria.io.fetch("maps/big_cluster.h5")

    input_map = maria.map.load(filename=map_filename,
                               nu=150e9, # in Hz
                               resolution=1/1024,
                               center=(150, 10),
                               frame="ra_dec",
                               units="Jy/pixel")

    input_map.to(units="uK_RJ").plot()


We can then add the map to a simulation as

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site,
                           plan=my_plan,
                           atmosphere="2d",
                           cmb="generate,
                           map=input_map)


.. _Time-evolving maps:

++++++++++++++++++
Time-evolving maps
++++++++++++++++++

``maria`` supports maps that evolve in time, which is useful for modeling *e.g.* solar observations.

.. code-block:: python

    time_evolving_sun_path = fetch("maps/sun.h5")
    input_map = maria.map.load(filename=time_evolving_sun_path, t=1.7e9 + np.linspace(0, 180, 16))
    plan = maria.Plan(start_time=1.7e9,
                      duration=180,
                      scan_center=np.degrees(input_map.center),
                      scan_options={"radius": 0.25})
    sim = maria.Simulation(plan=plan, map=input_map)
    tod = sim.run()
    tod.plot()

.. warning::
    If the observation overruns the start or end of the map in time, it will lead to some funky discontinuities.
