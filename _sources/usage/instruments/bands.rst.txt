#####
Bands
#####

.. code-block:: python

    from maria.instrument import Band

    my_band = Band(center=150, # in GHz
                   width=30, # in GHz
                   efficiency=0.5
                   sensitivity=1e-5
                   )
