####
Maps
####

A map is represented by a ``Map`` object. We can load some pre-defined ones with

.. code-block:: python

    from maria.io import fetch

    map_filename = fetch("maps/big_cluster.fits")

    input_map = maria.map.read_fits(filename=map_filename,
                                    index=1, # which index of the HDU to read
                                    nu=150., # in GHz
                                    resolution=1/1024,
                                    center=(150, 10),
                                    frame="ra_dec",
                                    units="Jy/pixel")

    input_map.to(units="K_RJ").plot()


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
