###############################
The cosmic microwave background
###############################

To add in the CMB (either as something to observe, or a noise source), we add

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site,
                           plan=my_plan,
                           atmosphere="2d",
                           cmb="generate")

This two-dimensional model is a simplified approximation of atmospheric turbulence (see `T. W. Morris et al. (2022) <https://arxiv.org/abs/2111.01319>`_) that lets us more efficiently  simulate atmospheric fluctuations at higher resolutions.
We can customize the CMB by specifying a set of ``cmb_kwargs`` to, e.g., increase the resolution of the generated map.

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site,
                           plan=my_plan,
                           atmosphere="2d",
                           cmb="generate",
                           cmb_kwargs={"nside": 4096})
