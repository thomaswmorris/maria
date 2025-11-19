
.. _cmb:

###
CMB
###

To add in the CMB (either as something to observe, or a noise source), we add

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site,
                           plan=my_plan,
                           atmosphere="2d",
                           cmb="generate")

We can customize the CMB by specifying a set of ``cmb_kwargs`` to, e.g., increase the resolution of the generated map.

.. code-block:: python

    sim = maria.Simulation(instrument=my_instrument,
                           site=my_site,
                           plan=my_plan,
                           atmosphere="2d",
                           cmb="generate",
                           cmb_kwargs={"nside": 4096})


###########
CMB patches
###########

To simulate CMB observations at high resolution, we can generate a small patch as a ``Map`` using

.. code-block:: python

    from maria.cmb import generate_cmb_patch

    cmb_patch = generate_cmb_patch(width=5) # in degrees

    cmb_patch.plot(cmap="cmb")

which can then be passed as an input map to a ``Simulation``.
