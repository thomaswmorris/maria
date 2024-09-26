###########
Simulations
###########

We can generate a realization of the CMB by running

.. code-block:: python

    cmb = maria.cmb.generate_cmb(**cmb_kwargs)

This is done automatically if we pass `cmb="generate"` when running a simulation.
