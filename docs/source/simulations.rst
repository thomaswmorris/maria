###########
Simulations
###########

Simulations in ``maria`` are done with a ``Simulation``, to which we pass an instrument, a site, and an observing plan::

    import maria

    sim = maria.Simulation(instrument="ACT", site="llano_de_chajnantor", plan="stare")


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
    act_sim = Simulation(array='ACT', site='cerro_toco', pointing='back-and-forth')

    # MUSTANG-2
    mustang2_sim = Simulation(array='MUSTANG-2', site='green_bank', pointing='daisy')

    # The South Pole Telescope
    spt_sim = Simulation(array='SPT', site='amundsen_scott', pointing='stare')
