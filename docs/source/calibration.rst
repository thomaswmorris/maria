###########
Simulations
###########

Simulations in ``maria`` are done with a ``Simulation``, to which we pass an instrument, a site, and an observing plan:

    import maria

    sim = maria.Simulation(instrument="ACT",
                           site="llano_de_chajnantor",
                           plan="one_minute_zenith_stare")

Running a simulation will spit out a ``TOD`` (short for time-ordered data)

    tod = sim.run()

which has the simulated timestreams for each pixel along with some metadata. The same simulation can produce any number of ``TOD`` objects, continuing from where the last TOD left off::

    another_tod = sim.run()

    yet_another_tod = sim.run()
