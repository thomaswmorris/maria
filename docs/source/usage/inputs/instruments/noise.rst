
.. _noise:
#####
Noise
#####

We can control the white noise level of the detectors in a band with either an explicit noise-equivalent power level (in Watts) as

.. code-block:: python

    from maria import Band

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5
                   NEP=1e-15)

or with an implicit Rayleigh-Jeans sensitivity as

.. code-block:: python

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5
                   NET_RJ=1e-5) # in K_RJ

or in CMB units as

.. code-block:: python

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5
                   NET_CMB=1e-5) # in K_CMB

Since the sensitivity on the sky in temperature units depends on both the instrument and the observing conditions (i.e. the atmosphere), this method computes an NEP assuming there is no atmosphere. To include the effects of atmospheric opacity in this conversion, we can supply some ``spectrum_kwargs`` as

.. code-block:: python

    my_band = Band(center=150e9, # in Hz
                   width=30e9, # in Hz
                   efficiency=0.5, # in K_RJ
                   spectrum_kwargs={"region": "chajnantor", 
                                    "zenith_pwv": 1e0, # in mm
                                    "elevation": 45}) # in degrees


+++++++++++++++
Non-white noise
+++++++++++++++

Bolometers are typically dominated by non-white noise below a certain frequency threshold (the "knee"). We can add in pink noise to our band by supplying a ``knee`` parameter (in Hz).
