

###########
Simulations
###########

++++++++++++++++++++
Creating simulations
++++++++++++++++++++

Simulations in ``maria`` are done with a ``Simulation``. Each ``Simulation`` always needs an ``instrument``, a ``site``, and a ``plan``:

.. code-block:: python

    import maria

    sim = maria.Simulation(instrument="ACT",
                           site="cerro_toco",
                           plan="stare")

Each of these string refers to a pre-defined configuration that ``maria`` knows about.

.. hint:: To see all available pre-defined instruments, sites, and plans, run ``print(maria.all_instruments)``, ``print(maria.all_sites)``, and ``print(maria.all_plans)``. For documentation on customizing these inputs, see the :ref:`instruments`, :ref:`sites`, and :ref:`plans` sections.

The simulation we instantiated above won't be very interesting, since we haven't given it anything to observe yet; if we ran it, we would just see detector noise.
We optionally give the ``Simulation`` an ``atmosphere``, a ``cmb``, or a ``map`` (or any combination), for example as

.. code-block:: python

    sim = maria.Simulation(instrument="ACT",
                           site="cerro_toco",
                           plan="stare",
                           atmosphere="2d",
                           cmb="generate",
                           map=input_map,
                           )

For documentation on these inputs (including how to customize them), see the :ref:`atmosphere`, :ref:`cmb`, and :ref:`maps` sections.

.. note:: While we add any combination of ``atmosphere``, ``cmb``, and ``map`` parameters, they are not independent; the presence of an atmosphere will affect a ``map`` and ``cmb``


+++++++++++++++++++
Running simulations
+++++++++++++++++++

Running a simulation will return the time-ordered data as a ``TOD``.

.. code-block:: python
    
    tod = sim.run()

Subsequent runs will each return another ``TOD`` where the last simulation left off

.. code-block:: python
    
    another_tod = sim.run()
    yet_another_tod = sim.run()

Each ``TOD`` can be fed into ``maria``'s native mapping code (see :ref:`mapping`) or exported as a FITS file to be used by some other package or software.
