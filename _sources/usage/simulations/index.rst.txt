########
Overview
########

Simulations in ``maria`` are done with a ``Simulation``, to which we pass an instrument, a site, and an observing plan::

    import maria

    sim = maria.Simulation(instrument="ACT", site="cerro_toco", plan="stare")


Running a simulation will spit out a ``TOD`` (short for time-ordered data)::

    tod = sim.run()

which has the simulated timestreams for each pixel along with some metadata. The same simulation can produce any number of ``TOD`` objects, continuing from where the last TOD left off::

    another_tod = sim.run()

    yet_another_tod = sim.run()



+++++++++++++++++++++++
Customizing simulations
+++++++++++++++++++++++

We can minimally customize a simulation by specifying an ``array`` (defining the instrument), a ``site`` (where on earth it is), and a ``pointing`` (defining how it observes). Simulations for a few different telescopes might be instantiated as::

    # The Atacama Cosmology Telescope
    act_sim = Simulation(instrument="ACT", site="cerro_toco", plan="back_and_forth")

    # MUSTANG-2
    mustang2_sim = Simulation(instrument="MUSTANG-2", site="green_bank", plan="daisy_2deg_60s")

    # The South Pole Telescope
    spt_sim = Simulation(instrument="SPT", site="amundsen_scott", plan="zenith_stare_60s")
