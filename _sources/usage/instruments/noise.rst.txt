#####
Noise
#####

Noise is specified for a given band

.. code-block:: python

    band_with_noise = Band(center=150,
                           width=30,
                           sensitivity=5e-5,
                           knee=1e0,
                           gain_error=5e-2)

We can also use the

.. code-block:: python

    band_with_noise = Band(center=150,
                           width=30,
                           sensitivity=5e-5,
                           knee=1e0,
                           gain_error=5e-2)


Note that any sensitivity units are only implicit due to the imperfect efficiency (caused by the instrument and the atmosphere).
For a heuristic, the band implicitly assumes that we are observing the zenith at the ALMA site with a PWV of 1mm, and converts the given sensitivity to the corresponding noise equivalent power.
We can customize this by setting the sensitivity as

.. code-block:: python

    my_band.set_sensitivity(1e-5, pwv=2, elevation=45, region="mauna_kea")
