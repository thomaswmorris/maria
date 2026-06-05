

###############
Getting started
###############

``maria`` is a meant to be a flexible, intuitive simulator that is capable of simulating time-ordered data for most astronomical observatories.

++++++++++++++++++++
Creating simulations
++++++++++++++++++++

Simulations in ``maria`` are done with a ``Simulation``. Each ``Simulation`` always minimally needs an ``instrument`` (something to observe with), a ``site`` (somewhere to observe from), and a ``plan`` (something to do). Each of these can be either a string (corresponding to a pre-defined ``maria`` input) or an ``Instrument``, ``Site``, or ``Plan`` object, for example as 

.. code-block:: python

    sim = Simulation(instrument="apex/saboca",
                     site="cerro_chajnantor"
                     plans="stare")

.. hint:: To see all available pre-defined instruments, sites, and plans, run ``print(maria.all_instruments)``, ``print(maria.all_sites)``, and ``print(maria.all_plans)``.

or a custom object encoding all the parameters of your custom simulation. 

.. code-block:: python

    from maria import Simulation

    sim = Simulation(instrument=my_instrument,
                     site=my_site
                     plans=my_plans)

For documentation on creating these object, refer to the :ref:`instruments`, :ref:`sites`, and :ref:`plans` sections.


+++++++
Sources
+++++++

The simulations above have nothing to observe; if we ran them, our time-ordered data consist only of instrument noise. We can optionally give it at source in an ``atmosphere``, a ``map``, or a ``cmb`` (or all at once), for example as

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site
                           plan=my_plan,
                           atmosphere=my_atmosphere,
                           cmb=my_cmb,
                           map=my_map)

.. note:: While we add any combination of ``atmosphere``, ``cmb``, and ``map`` parameters, they are not independent; the presence of an atmosphere will affect a ``map`` and ``cmb``

For documentation on these inputs (including how to customize them), see the :ref:`atmosphere`, :ref:`cmb`, and :ref:`maps` sections.



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

Each ``TOD`` can be fed into ``maria``'s native mapping code (see :ref:`mapping`) or exported as a file to be used by some other package or software.

For more documentation on working with a ``TOD``, see the :ref:`tod` sections.
