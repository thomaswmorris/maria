##########
Atmosphere
##########

To add in atmosphere to the simulation, we add in an atmospheric model.

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site,
                           plan=my_plan,
                           atmosphere="2d")

This two-dimensional model is a simplified approximation of atmospheric turbulence (see `T. W. Morris et al. (2022) <https://arxiv.org/abs/2111.01319>`_) that lets us more efficiently  simulate atmospheric fluctuations at higher resolutions.
``maria`` generates an atmosphere based on the weather parameters from the simulation's ``Site``. The atmosphere both emits radiation and blocks light from celestial sources.
