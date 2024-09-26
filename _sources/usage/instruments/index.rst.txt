########
Overview
########

The observing instrument is instantiated as an ``Instrument``.
The simplest way to get an instrument is to grab a pre-defined one, e.g.:

.. code-block:: python

    # The Atacama Cosmology Telescope
    act = maria.get_instrument('ACT')

    # The Atacama Large Millimeter Array
    alma = maria.get_instrument('ALMA')

    # MUSTANG-2
    m2 = maria.get_instrument('MUSTANG-2')


To see the list of supported instruments, run

.. code-block:: python

    print(maria.all_instruments)
